from hashlib import sha256
from pathlib import Path
from typing import Any

import networkx as nx

from .config import Phase1Config
from .evaluation import (
    best_candidate,
    dataset_fingerprint,
    evaluate_bound_split,
    evaluate_split,
)
from .known_invariants import compute_known_invariant_values
from .logging_io import write_json
from .sandbox import SandboxEvaluator
from .scoring import compute_novelty_ci
from .targets import target_values
from .types import CheckpointState


def _extract_spearman(metrics: dict[str, object] | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("spearman")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _baseline_health(
    baseline_results: dict[str, object] | None,
) -> tuple[bool, bool, str, float | None, float | None]:
    if not isinstance(baseline_results, dict):
        return False, False, "missing", None, None
    pysr_payload = baseline_results.get("pysr_baseline")
    stat_payload = baseline_results.get("stat_baselines")

    if isinstance(pysr_payload, dict):
        pysr_status = str(pysr_payload.get("status", "missing"))
        pysr_val_spearman = _extract_spearman(pysr_payload.get("val_metrics"))
        pysr_test_spearman = _extract_spearman(pysr_payload.get("test_metrics"))
    else:
        pysr_status = "missing"
        pysr_val_spearman = None
        pysr_test_spearman = None

    stat_ok = False
    if isinstance(stat_payload, dict):
        stat_ok = any(
            isinstance(result, dict) and str(result.get("status")) == "ok"
            for result in stat_payload.values()
        )

    pysr_ok = pysr_status == "ok"
    return stat_ok, pysr_ok, pysr_status, pysr_val_spearman, pysr_test_spearman


def _novelty_ci_for_split(
    code: str,
    split_features: list[dict[str, Any]],
    known_values: dict[str, list[float]],
    seed_offset: int,
    cfg: Phase1Config,
    evaluator: SandboxEvaluator,
) -> dict[str, object]:
    y_pred_raw = evaluator.evaluate(code, split_features)
    valid_pairs = [(idx, yp) for idx, yp in enumerate(y_pred_raw) if yp is not None]
    if not valid_pairs:
        return {
            "max_ci_upper_abs_rho": 0.0,
            "novelty_passed": False,
            "threshold": cfg.novelty_threshold,
            "per_invariant": {},
        }
    valid_indices, y_pred_valid = zip(*valid_pairs, strict=True)
    known_subset = {
        name: [values[idx] for idx in valid_indices] for name, values in known_values.items()
    }
    return compute_novelty_ci(
        candidate_values=list(y_pred_valid),
        known_invariants=known_subset,
        n_bootstrap=cfg.novelty_bootstrap_samples,
        seed=cfg.seed + seed_offset,
        novelty_threshold=cfg.novelty_threshold,
    )


def write_phase1_summary(
    cfg: Phase1Config,
    state: CheckpointState,
    evaluator: SandboxEvaluator,
    artifacts_dir: Path,
    stop_reason: str,
    features_train: list[dict[str, Any]],
    features_val: list[dict[str, Any]],
    features_test: list[dict[str, Any]],
    features_sanity: list[dict[str, Any]],
    datasets_val: list[nx.Graph],
    datasets_test: list[nx.Graph],
    datasets_sanity: list[nx.Graph],
    y_true_train: list[float],
    y_true_val: list[float],
    y_true_test: list[float],
    baseline_results: dict[str, object] | None,
    self_correction_stats: dict[str, Any],
) -> None:
    best = best_candidate(state)
    summary_path = artifacts_dir / "phase1_summary.json"
    if best is None:
        payload = {
            "experiment_id": state.experiment_id,
            "success": False,
            "reason": "no_candidates",
            "stop_reason": stop_reason,
        }
        write_json(payload, summary_path)
        return

    known_val = compute_known_invariant_values(
        datasets_val,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    known_test = compute_known_invariant_values(
        datasets_test,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    y_sanity = target_values(datasets_sanity, cfg.target_name)
    val_metrics = evaluate_split(best.code, features_val, y_true_val, cfg, evaluator, known_val)
    test_metrics = evaluate_split(
        best.code, features_test, y_true_test, cfg, evaluator, known_test
    )
    train_metrics = evaluate_split(best.code, features_train, y_true_train, cfg, evaluator)
    sanity_metrics = evaluate_split(best.code, features_sanity, y_sanity, cfg, evaluator)

    novelty_ci = {
        "validation": _novelty_ci_for_split(
            best.code, features_val, known_val, 17, cfg, evaluator
        ),
        "test": _novelty_ci_for_split(
            best.code, features_test, known_test, 29, cfg, evaluator
        ),
    }

    is_bounds = cfg.fitness_mode in ("upper_bound", "lower_bound")

    if is_bounds:
        val_bound_metrics = evaluate_bound_split(
            best.code,
            features_val,
            y_true_val,
            evaluator,
            fitness_mode=cfg.fitness_mode,
            tolerance=cfg.bound_tolerance,
        )
        test_bound_metrics = evaluate_bound_split(
            best.code,
            features_test,
            y_true_test,
            evaluator,
            fitness_mode=cfg.fitness_mode,
            tolerance=cfg.bound_tolerance,
        )
        bounds_success = (
            val_bound_metrics["bound_score"] >= cfg.success_bound_score_threshold
            and val_bound_metrics["satisfaction_rate"] >= cfg.success_satisfaction_threshold
        )
        success = bounds_success
        success_criteria = None
        success_criteria_bounds = {
            "bound_score_threshold": cfg.success_bound_score_threshold,
            "satisfaction_threshold": cfg.success_satisfaction_threshold,
            "val_bound_score": val_bound_metrics["bound_score"],
            "val_satisfaction_rate": val_bound_metrics["satisfaction_rate"],
            "passed": bounds_success,
        }
        baseline_comparison = None
        schema_version = 4
    else:
        stat_ok, pysr_ok, pysr_status, pysr_val_spearman, pysr_test_spearman = _baseline_health(
            baseline_results
        )
        candidate_val_spearman = float(val_metrics.get("spearman", 0.0))
        candidate_test_spearman = float(test_metrics.get("spearman", 0.0))

        threshold_passed = abs(candidate_val_spearman) >= cfg.success_spearman_threshold
        baselines_available = baseline_results is not None
        baselines_healthy = stat_ok or pysr_ok
        baselines_passed = (not cfg.require_baselines_for_success) or (
            baselines_available and baselines_healthy
        )
        pysr_parity_passed = True
        pysr_parity_reason = "disabled"
        if cfg.enforce_pysr_parity_for_success:
            if pysr_status != "ok" or pysr_val_spearman is None:
                pysr_parity_passed = False
                pysr_parity_reason = "pysr_missing_or_unavailable"
            else:
                pysr_parity_passed = (
                    candidate_val_spearman + cfg.pysr_parity_epsilon >= pysr_val_spearman
                )
                pysr_parity_reason = "ok"

        success_criteria = {
            "success_spearman_threshold": cfg.success_spearman_threshold,
            "threshold_passed": threshold_passed,
            "require_baselines_for_success": cfg.require_baselines_for_success,
            "baselines_available": baselines_available,
            "baselines_healthy": baselines_healthy,
            "stat_baseline_ok": stat_ok,
            "pysr_ok": pysr_ok,
            "baselines_passed": baselines_passed,
            "enforce_pysr_parity_for_success": cfg.enforce_pysr_parity_for_success,
            "pysr_parity_epsilon": cfg.pysr_parity_epsilon,
            "pysr_status": pysr_status,
            "candidate_val_spearman": candidate_val_spearman,
            "pysr_val_spearman": pysr_val_spearman,
            "pysr_parity_passed": pysr_parity_passed,
            "pysr_parity_reason": pysr_parity_reason,
        }
        success = threshold_passed and baselines_passed and pysr_parity_passed

        baseline_comparison = {
            "candidate": {
                "val_spearman": candidate_val_spearman,
                "test_spearman": candidate_test_spearman,
            },
            "pysr": {
                "status": pysr_status,
                "val_spearman": pysr_val_spearman,
                "test_spearman": pysr_test_spearman,
            },
        }
        val_bound_metrics = None
        test_bound_metrics = None
        success_criteria_bounds = None
        schema_version = 3

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "experiment_id": state.experiment_id,
        "fitness_mode": cfg.fitness_mode,
        "model_name": cfg.model_name,
        "best_candidate_id": best.id,
        "best_candidate_code_sha256": sha256(best.code.encode("utf-8")).hexdigest(),
        "best_val_score": state.best_val_score,
        "stop_reason": stop_reason,
        "success": success,
        "config": cfg.to_dict(),
        "final_generation": state.generation,
        "island_candidate_counts": {
            str(island_id): len(candidates) for island_id, candidates in state.islands.items()
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sanity_metrics": sanity_metrics,
        "novelty_ci": novelty_ci,
        "dataset_fingerprint": dataset_fingerprint(
            cfg, y_true_train, y_true_val, y_true_test
        ),
        "self_correction_stats": self_correction_stats,
    }
    if baseline_comparison is not None:
        payload["baseline_comparison"] = baseline_comparison
    if success_criteria is not None:
        payload["success_criteria"] = success_criteria
    if is_bounds:
        payload["bounds_metrics"] = {"val": val_bound_metrics, "test": test_bound_metrics}
        payload["success_criteria_bounds"] = success_criteria_bounds
    if cfg.persist_candidate_code_in_summary:
        payload["best_candidate_code"] = best.code
    write_json(payload, summary_path)
