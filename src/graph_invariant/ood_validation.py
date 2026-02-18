"""Out-of-distribution validation for discovered graph invariant formulas.

Evaluates a candidate formula on graphs that differ from the training
distribution (n in [30,100], ER/BA/WS/RGG/SBM) in three ways:
  - large_random: same types but n in [200,500]
  - extreme_params: extreme densities/degrees, n in [50,200]
  - special_topology: deterministic structures (barbell, grid, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .data import (
    generate_ood_extreme_params,
    generate_ood_large_random,
    generate_ood_special_topology,
)
from .known_invariants import compute_feature_dicts
from .logging_io import write_json
from .sandbox import SandboxEvaluator
from .scoring import compute_bound_metrics, compute_metrics
from .targets import target_values

logger = logging.getLogger(__name__)


# ── Dataset generation ───────────────────────────────────────────────


def generate_ood_datasets(
    seed: int = 42,
    num_large: int = 100,
    num_extreme: int = 50,
) -> dict[str, list[nx.Graph]]:
    """Generate the three OOD dataset categories.

    Returns a dict mapping category name to list of graphs:
      - large_random: same types as training but n in [200, 500]
      - extreme_params: extreme densities/degrees, n in [50, 200]
      - special_topology: deterministic structures (barbell, grid, etc.)
    """
    rng = np.random.default_rng(seed)
    return {
        "large_random": generate_ood_large_random(rng, num_large),
        "extreme_params": generate_ood_extreme_params(rng, num_extreme),
        "special_topology": generate_ood_special_topology(),
    }


# ── Evaluation ───────────────────────────────────────────────────────


def _evaluate_ood_split(
    code: str,
    graphs: list[nx.Graph],
    target_name: str,
    evaluator: Any,
    fitness_mode: str = "correlation",
    tolerance: float = 1e-9,
    enable_spectral_feature_pack: bool = True,
) -> dict[str, float | int]:
    """Evaluate candidate code against a list of graphs."""
    features = compute_feature_dicts(
        graphs,
        include_spectral_feature_pack=enable_spectral_feature_pack,
    )
    y_true = target_values(graphs, target_name)
    y_pred_raw = evaluator.evaluate(code, features)
    valid_pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred_raw, strict=True) if yp is not None]
    if not valid_pairs:
        return {"valid_count": 0, "total_count": len(graphs)}

    y_t, y_p = zip(*valid_pairs, strict=True)
    result: dict[str, float | int] = {
        "valid_count": len(valid_pairs),
        "total_count": len(graphs),
    }
    if fitness_mode in ("upper_bound", "lower_bound"):
        bm = compute_bound_metrics(list(y_t), list(y_p), mode=fitness_mode, tolerance=tolerance)
        result["bound_score"] = bm.bound_score
        result["satisfaction_rate"] = bm.satisfaction_rate
        result["mean_gap"] = bm.mean_gap
    else:
        metrics = compute_metrics(list(y_t), list(y_p))
        result["spearman"] = metrics.rho_spearman
        result["pearson"] = metrics.r_pearson
        result["rmse"] = metrics.rmse
        result["mae"] = metrics.mae
    return result


# ── Main entry point ─────────────────────────────────────────────────


def run_ood_validation(
    summary_path: str,
    output_dir: str,
    seed: int = 42,
    num_large: int = 100,
    num_extreme: int = 50,
) -> int:
    """Evaluate best candidate from phase1_summary on OOD graphs.

    Returns 0 on success, 1 on error.
    """
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    code = summary.get("best_candidate_code")
    if not code:
        logger.error(
            "summary missing 'best_candidate_code'. "
            "Re-run phase1 with persist_candidate_code_in_summary=true."
        )
        return 1

    cfg = summary.get("config", {})
    target_name = cfg.get("target_name", "average_shortest_path_length")
    fitness_mode = cfg.get("fitness_mode", "correlation")
    timeout_sec = cfg.get("timeout_sec", 2.0)
    memory_mb = cfg.get("memory_mb", 256)
    enable_spectral_feature_pack = bool(cfg.get("enable_spectral_feature_pack", True))

    datasets = generate_ood_datasets(seed=seed, num_large=num_large, num_extreme=num_extreme)

    results: dict[str, Any] = {}
    with SandboxEvaluator(timeout_sec=timeout_sec, memory_mb=memory_mb) as evaluator:
        for category, graphs in datasets.items():
            results[category] = _evaluate_ood_split(
                code=code,
                graphs=graphs,
                target_name=target_name,
                evaluator=evaluator,
                fitness_mode=fitness_mode,
                enable_spectral_feature_pack=enable_spectral_feature_pack,
            )

    out = Path(output_dir)
    write_json(results, out / "ood_validation.json")
    return 0
