"""Out-of-distribution validation for discovered graph invariant formulas.

Evaluates a candidate formula on graphs that differ from the training
distribution (n in [30,100], ER/BA/WS/RGG/SBM) in three ways:
  - large_random: same types but n in [200,500]
  - extreme_params: extreme densities/degrees, n in [50,200]
  - special_topology: deterministic structures (barbell, grid, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .data import connected_subgraph, generate_graph
from .known_invariants import compute_feature_dicts
from .logging_io import write_json
from .sandbox import SandboxEvaluator
from .scoring import compute_bound_metrics, compute_metrics

# ── Target functions (imported lazily to avoid circular deps) ────────

_TARGET_FUNCTIONS: dict[str, Any] | None = None


def _get_target_functions() -> dict[str, Any]:
    global _TARGET_FUNCTIONS  # noqa: PLW0603
    if _TARGET_FUNCTIONS is None:
        from .cli import TARGET_FUNCTIONS

        _TARGET_FUNCTIONS = TARGET_FUNCTIONS
    return _TARGET_FUNCTIONS


def _target_values(graphs: list[nx.Graph], target_name: str) -> list[float]:
    fns = _get_target_functions()
    fn = fns.get(target_name)
    if fn is None:
        raise ValueError(f"unsupported target: {target_name}")
    return [fn(g) for g in graphs]


# ── Dataset generation ───────────────────────────────────────────────


def _generate_large_random(rng: np.random.Generator, count: int) -> list[nx.Graph]:
    """Same graph types as training but with n in [200, 500]."""
    graphs: list[nx.Graph] = []
    for _ in range(count):
        n = int(rng.integers(200, 501))
        graphs.append(generate_graph(rng, n))
    return graphs


def _generate_extreme_params(rng: np.random.Generator, count: int) -> list[nx.Graph]:
    """Graphs with extreme densities/degrees, n in [50, 200]."""
    graphs: list[nx.Graph] = []
    for _ in range(count):
        n = int(rng.integers(50, 201))
        kind = rng.choice(["dense_er", "sparse_er", "high_ba", "low_ws"])
        if kind == "dense_er":
            g = nx.erdos_renyi_graph(n, float(rng.uniform(0.4, 0.7)), seed=rng)
        elif kind == "sparse_er":
            g = nx.erdos_renyi_graph(n, float(rng.uniform(0.01, 0.04)), seed=rng)
        elif kind == "high_ba":
            m = int(rng.integers(8, min(15, n - 1) + 1))
            g = nx.barabasi_albert_graph(n, m, seed=rng)
        else:  # low_ws
            k = 2
            g = nx.watts_strogatz_graph(n, k, float(rng.uniform(0.01, 0.1)), seed=rng)
        graphs.append(connected_subgraph(g))
    return graphs


def _generate_special_topology() -> list[nx.Graph]:
    """Deterministic structures for structural generalization."""
    graphs: list[nx.Graph] = []
    # Barbell
    graphs.append(nx.barbell_graph(20, 5))
    # Grid
    graphs.append(nx.grid_2d_graph(8, 8))
    # Circular ladder
    graphs.append(nx.circular_ladder_graph(30))
    # Random regular (deterministic seed)
    graphs.append(nx.random_regular_graph(4, 50, seed=0))
    # Powerlaw cluster
    graphs.append(nx.powerlaw_cluster_graph(100, 3, 0.5, seed=0))
    # Sanity set
    graphs.append(nx.karate_club_graph())
    graphs.append(nx.les_miserables_graph())
    graphs.append(nx.florentine_families_graph())
    # Ensure all connected and relabeled
    return [connected_subgraph(g) for g in graphs]


def generate_ood_datasets(
    seed: int = 42,
    num_large: int = 100,
    num_extreme: int = 50,
) -> dict[str, list[nx.Graph]]:
    """Generate the three OOD dataset categories."""
    rng = np.random.default_rng(seed)
    return {
        "large_random": _generate_large_random(rng, num_large),
        "extreme_params": _generate_extreme_params(rng, num_extreme),
        "special_topology": _generate_special_topology(),
    }


# ── Evaluation ───────────────────────────────────────────────────────


def _evaluate_ood_split(
    code: str,
    graphs: list[nx.Graph],
    target_name: str,
    evaluator: Any,
    fitness_mode: str = "correlation",
    tolerance: float = 1e-9,
) -> dict[str, float | int]:
    """Evaluate candidate code against a list of graphs."""
    features = compute_feature_dicts(graphs)
    y_true = _target_values(graphs, target_name)
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
        print(
            "ERROR: summary missing 'best_candidate_code'. "
            "Re-run phase1 with persist_candidate_code_in_summary=true."
        )
        return 1

    cfg = summary.get("config", {})
    target_name = cfg.get("target_name", "average_shortest_path_length")
    fitness_mode = cfg.get("fitness_mode", "correlation")
    timeout_sec = cfg.get("timeout_sec", 2.0)
    memory_mb = cfg.get("memory_mb", 256)

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
            )

    out = Path(output_dir)
    write_json(results, out / "ood_validation.json")
    return 0
