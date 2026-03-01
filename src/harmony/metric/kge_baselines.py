"""Unified KGE baseline runner for all embedding models.

Provides a single entry point to evaluate a KG with all available KGE models
(DistMult, TransE, RotatE, ComplEx) using the same protocol.

Also defines conceptual baselines:
  - LLM-only: proposals accepted by validity check alone (no Harmony scoring)
  - No-QD: greedy selection instead of MAP-Elites (single best proposal kept)

These are search-loop configurations, not metric-level functions.  The
run_all_kge_baselines() function handles the metric-level comparison.
"""

from __future__ import annotations

from harmony.types import KnowledgeGraph


def run_all_kge_baselines(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> dict[str, float]:
    """Evaluate KG with all KGE models. Returns model_name → Hits@K.

    Models: random, frequency, distmult, transe, rotate, complex.
    """
    from harmony.metric.baselines import (
        baseline_complex,
        baseline_distmult_alone,
        baseline_frequency,
        baseline_random,
        baseline_rotate,
        baseline_transe,
    )

    kw = dict(seed=seed, mask_ratio=mask_ratio, k=k)
    kw_kge = dict(**kw, dim=dim, n_epochs=n_epochs)

    return {
        "random": baseline_random(kg, **kw),
        "frequency": baseline_frequency(kg, **kw),
        "distmult": baseline_distmult_alone(kg, **kw_kge),
        "transe": baseline_transe(kg, **kw_kge),
        "rotate": baseline_rotate(kg, **kw_kge),
        "complex": baseline_complex(kg, **kw_kge),
    }


# ---------------------------------------------------------------------------
# Conceptual baselines (search-loop level, not metric-level)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Conceptual baseline configurations for harmony_loop experiments.
#
# These are NOT Phase1Config instances — they document the keyword arguments
# to pass to ``run_harmony_loop()`` when running ablation baselines.
# ``alpha/beta/gamma/delta`` map to HarmonyConfig weights; ``accept_all_valid``,
# ``num_bins``, and ``greedy`` map to HarmonySearchState / MAP-Elites settings.
# ---------------------------------------------------------------------------

LLM_ONLY_CONFIG: dict[str, object] = {
    "description": "LLM-only: proposals accepted by validity alone, no Harmony",
    "alpha": 0.0,
    "beta": 0.0,
    "gamma": 0.0,
    "delta": 0.0,
    "accept_all_valid": True,
}

NO_QD_CONFIG: dict[str, object] = {
    "description": "No-QD: greedy selection, no MAP-Elites diversity",
    "num_bins": 1,
    "greedy": True,
}
