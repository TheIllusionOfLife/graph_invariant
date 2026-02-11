import ast
import math

import numpy as np
import scipy.stats
import sympy

from .types import EvaluationResult


def _nan_to_zero(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return float(value)


def compute_metrics(y_true: list[float], y_pred: list[float]) -> EvaluationResult:
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    if true_arr.shape != pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    rho, _ = scipy.stats.spearmanr(true_arr, pred_arr)
    pearson, _ = scipy.stats.pearsonr(true_arr, pred_arr)
    rmse = float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))
    mae = float(np.mean(np.abs(true_arr - pred_arr)))
    return EvaluationResult(
        rho_spearman=_nan_to_zero(float(rho)),
        r_pearson=_nan_to_zero(float(pearson)),
        rmse=rmse,
        mae=mae,
        valid_count=len(true_arr),
        error_count=0,
    )


def _extract_return_expression(code: str) -> str:
    try:
        module = ast.parse(code)
        for node in ast.walk(module):
            if isinstance(node, ast.Return) and node.value is not None:
                return ast.unparse(node.value)
    except Exception:
        return code
    return code


def compute_simplicity_score(code: str, w1: float = 0.5, w2: float = 0.5) -> float:
    ast_nodes = max(1, len(list(ast.walk(ast.parse(code)))))
    expr = _extract_return_expression(code)
    try:
        simplified = sympy.simplify(expr)
        expr_len = max(1, len(str(simplified)))
    except Exception:
        expr_len = max(1, len(expr))

    ast_term = 1.0 / (1.0 + math.log2(ast_nodes))
    expr_term = 1.0 / (1.0 + math.log2(expr_len))
    return (w1 * ast_term) + (w2 * expr_term)


def compute_novelty_bonus(
    candidate_values: list[float],
    known_invariants: dict[str, list[float]],
) -> float:
    if not known_invariants:
        return 1.0

    max_abs_rho = 0.0
    for values in known_invariants.values():
        rho, _ = scipy.stats.spearmanr(candidate_values, values)
        max_abs_rho = max(max_abs_rho, abs(_nan_to_zero(float(rho))))
    return max(0.0, 1.0 - max_abs_rho)


def compute_total_score(
    abs_spearman: float,
    simplicity: float,
    novelty_bonus: float,
    alpha: float = 0.6,
    beta: float = 0.2,
    gamma: float = 0.2,
) -> float:
    return (alpha * abs_spearman) + (beta * simplicity) + (gamma * novelty_bonus)
