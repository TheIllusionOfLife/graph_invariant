import ast
import math
import warnings

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

    if len(true_arr) == 0:
        return EvaluationResult(
            rho_spearman=0.0,
            r_pearson=0.0,
            rmse=0.0,
            mae=0.0,
            valid_count=0,
            error_count=0,
        )

    if len(true_arr) < 2:
        rho = 0.0
        pearson = 0.0
    else:
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


def _bootstrap_abs_spearman_ci_upper(
    candidate_values: np.ndarray,
    known_values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if candidate_values.shape != known_values.shape:
        raise ValueError("candidate_values and known_values must have the same shape")
    if candidate_values.size < 2:
        return 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.stats.ConstantInputWarning)
        point_rho, _ = scipy.stats.spearmanr(candidate_values, known_values)
    point_abs_rho = abs(_nan_to_zero(float(point_rho)))
    sample_size = candidate_values.size
    bootstrap_abs_rhos: list[float] = []
    all_indices = rng.integers(0, sample_size, size=(n_bootstrap, sample_size))
    for sample_idx in all_indices:
        sampled_candidate = candidate_values[sample_idx]
        sampled_known = known_values[sample_idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=scipy.stats.ConstantInputWarning)
            sampled_rho, _ = scipy.stats.spearmanr(sampled_candidate, sampled_known)
        bootstrap_abs_rhos.append(abs(_nan_to_zero(float(sampled_rho))))
    upper = float(np.quantile(np.asarray(bootstrap_abs_rhos, dtype=float), 0.95))
    return point_abs_rho, upper


def compute_novelty_ci(
    candidate_values: list[float],
    known_invariants: dict[str, list[float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    novelty_threshold: float = 0.7,
) -> dict[str, object]:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not known_invariants:
        return {
            "max_ci_upper_abs_rho": 0.0,
            "novelty_passed": True,
            "threshold": novelty_threshold,
            "per_invariant": {},
        }

    candidate_arr = np.asarray(candidate_values, dtype=float)
    rng = np.random.default_rng(seed)
    per_invariant: dict[str, dict[str, float]] = {}
    max_upper = 0.0
    for name, values in known_invariants.items():
        known_arr = np.asarray(values, dtype=float)
        point_abs_rho, ci_upper = _bootstrap_abs_spearman_ci_upper(
            candidate_values=candidate_arr,
            known_values=known_arr,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        max_upper = max(max_upper, ci_upper)
        per_invariant[name] = {
            "point_abs_rho": point_abs_rho,
            "ci_upper_abs_rho": ci_upper,
        }

    return {
        "max_ci_upper_abs_rho": max_upper,
        "novelty_passed": max_upper < novelty_threshold,
        "threshold": novelty_threshold,
        "per_invariant": per_invariant,
    }
