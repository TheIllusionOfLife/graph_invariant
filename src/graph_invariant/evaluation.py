import json
from hashlib import sha256
from typing import Any

from .config import Phase1Config
from .sandbox import SandboxEvaluator
from .scoring import compute_bound_metrics, compute_metrics, compute_novelty_bonus
from .types import Candidate, CheckpointState


def best_candidate(state: CheckpointState) -> Candidate | None:
    all_candidates = [candidate for island in state.islands.values() for candidate in island]
    if not all_candidates:
        return None
    return max(all_candidates, key=lambda c: c.val_score)


def global_best_score(state: CheckpointState) -> float:
    scores = [
        candidate.val_score for candidates in state.islands.values() for candidate in candidates
    ]
    if not scores:
        return 0.0
    return max(scores)


def evaluate_split(
    code: str,
    features_list: list[dict[str, Any]],
    y_true: list[float],
    cfg: Phase1Config,
    evaluator: SandboxEvaluator,
    known_invariants: dict[str, list[float]] | None = None,
) -> dict[str, float | int | None]:
    metrics, _, _ = evaluate_split_with_predictions(
        code=code,
        features_list=features_list,
        y_true=y_true,
        cfg=cfg,
        evaluator=evaluator,
        known_invariants=known_invariants,
    )
    return metrics


def evaluate_split_with_predictions(
    code: str,
    features_list: list[dict[str, Any]],
    y_true: list[float],
    cfg: Phase1Config,
    evaluator: SandboxEvaluator,
    known_invariants: dict[str, list[float]] | None = None,
) -> tuple[dict[str, float | int | None], list[int], list[float]]:
    y_pred_raw = evaluator.evaluate(code, features_list)
    valid_pairs = [
        (idx, yt, yp)
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred_raw, strict=True))
        if yp is not None
    ]
    if not valid_pairs:
        return (
            {
                "spearman": 0.0,
                "pearson": 0.0,
                "rmse": 0.0,
                "mae": 0.0,
                "valid_count": 0,
                "novelty_bonus": None,
            },
            [],
            [],
        )

    valid_indices, y_true_valid, y_pred_valid = zip(*valid_pairs, strict=True)
    metrics = compute_metrics(list(y_true_valid), list(y_pred_valid))
    novelty = None
    if known_invariants is not None:
        known_subset = {
            name: [values[idx] for idx in valid_indices]
            for name, values in known_invariants.items()
        }
        novelty = compute_novelty_bonus(list(y_pred_valid), known_subset)

    return (
        {
            "spearman": metrics.rho_spearman,
            "pearson": metrics.r_pearson,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "valid_count": metrics.valid_count,
            "novelty_bonus": novelty,
        },
        list(valid_indices),
        list(y_pred_valid),
    )


def evaluate_bound_split(
    code: str,
    features_list: list[dict[str, Any]],
    y_true: list[float],
    evaluator: SandboxEvaluator,
    fitness_mode: str,
    tolerance: float = 1e-9,
) -> dict[str, float | int]:
    """Evaluate a candidate against a data split in bounds mode."""
    y_pred_raw = evaluator.evaluate(code, features_list)
    valid_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred_raw, strict=True) if yp is not None]
    if not valid_pairs:
        return {
            "bound_score": 0.0,
            "satisfaction_rate": 0.0,
            "mean_gap": 0.0,
            "violation_count": 0,
            "valid_count": 0,
        }
    y_t, y_p = zip(*valid_pairs, strict=True)
    bm = compute_bound_metrics(list(y_t), list(y_p), mode=fitness_mode, tolerance=tolerance)
    return {
        "bound_score": bm.bound_score,
        "satisfaction_rate": bm.satisfaction_rate,
        "mean_gap": bm.mean_gap,
        "violation_count": bm.violation_count,
        "valid_count": bm.valid_count,
    }


def dataset_fingerprint(
    cfg: Phase1Config, y_train: list[float], y_val: list[float], y_test: list[float]
) -> str:
    payload = {
        "seed": cfg.seed,
        "target_name": cfg.target_name,
        "model_name": cfg.model_name,
        "num_train_graphs": cfg.num_train_graphs,
        "num_val_graphs": cfg.num_val_graphs,
        "num_test_graphs": cfg.num_test_graphs,
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return sha256(text.encode("utf-8")).hexdigest()
