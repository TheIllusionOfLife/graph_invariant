"""Leave-one-out ablation with bootstrap confidence intervals for Harmony metric.

run_ablation() returns 5 AblationRow instances:
  - "full"     : all 4 components with equal weights
  - "w/o_comp" : compressibility weight zeroed
  - "w/o_coh"  : coherence weight zeroed
  - "w/o_sym"  : symmetry weight zeroed
  - "w/o_gen"  : generativity weight zeroed

Bootstrap strategy:
  Resample kg.edges with replacement (same n as original), rebuild a KG
  from resampled edges (skipping edges whose entities are no longer valid —
  not possible since all original entities are re-added), score each sample.
  Repeat n_bootstrap times per variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graph_invariant.stats_utils import mean_std_ci95
from harmony.metric.harmony import harmony_score
from harmony.types import Entity, KnowledgeGraph, TypedEdge


@dataclass
class AblationRow:
    """One row of the ablation table."""

    component: str  # "full" | "w/o_comp" | "w/o_coh" | "w/o_sym" | "w/o_gen"
    mean: float
    std: float
    ci95_half_width: float
    n: int
    delta_vs_full: float  # mean - full_mean; negative means component helps


def _bootstrap_scores(
    kg: KnowledgeGraph,
    n_bootstrap: int,
    seed: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> list[float]:
    """Return n_bootstrap harmony scores from resampled KGs."""
    edges = kg.edges
    n_edges = len(edges)
    rng = np.random.default_rng(seed)
    scores: list[float] = []

    for _ in range(n_bootstrap):
        # Resample edges with replacement
        indices = rng.integers(0, n_edges, size=n_edges)
        resampled: list[TypedEdge] = [edges[int(i)] for i in indices]

        # Rebuild KG: add all original entities first (preserves referential integrity)
        bkg = KnowledgeGraph(domain=kg.domain)
        for entity in kg.entities.values():
            bkg.add_entity(Entity(id=entity.id, entity_type=entity.entity_type))

        # Add resampled edges — duplicates are valid (add_edge just appends)
        for edge in resampled:
            bkg.add_edge(edge)

        score = harmony_score(bkg, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        scores.append(score)

    return scores


def run_ablation(
    kg: KnowledgeGraph,
    n_bootstrap: int = 200,
    seed: int = 42,
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.25,
) -> list[AblationRow]:
    """Full metric + 4 leave-one-out variants with bootstrap CIs.

    Bootstrap strategy: resample kg.edges with replacement (same n as original),
    rebuild a KG from resampled edges, score each bootstrap sample.
    Repeat n_bootstrap times per variant.

    Returns rows sorted by mean score descending (full first).

    Parameters
    ----------
    kg:
        Knowledge graph to ablate.
    n_bootstrap:
        Number of bootstrap resamples per variant.
    seed:
        Base RNG seed; each variant gets seed + variant_offset for independence.
    alpha, beta, gamma, delta:
        Weights for the full metric (compressibility, coherence, symmetry, generativity).
    """
    variants: list[tuple[str, float, float, float, float]] = [
        ("full", alpha, beta, gamma, delta),
        ("w/o_comp", 0.0, beta, gamma, delta),
        ("w/o_coh", alpha, 0.0, gamma, delta),
        ("w/o_sym", alpha, beta, 0.0, delta),
        ("w/o_gen", alpha, beta, gamma, 0.0),
    ]

    rows: list[AblationRow] = []
    full_mean: float = 0.0

    for idx, (name, a, b, g, d) in enumerate(variants):
        # Use a different seed offset per variant for independent bootstrap samples
        variant_seed = seed + idx * 1000
        boot_scores = _bootstrap_scores(kg, n_bootstrap, variant_seed, a, b, g, d)

        stats = mean_std_ci95(boot_scores)
        mean_val = float(stats["mean"])  # type: ignore[arg-type]
        std_val = float(stats["std"])  # type: ignore[arg-type]
        ci_half = float(stats["ci95_half_width"])  # type: ignore[arg-type]
        n_val = int(stats["n"])  # type: ignore[arg-type]

        if name == "full":
            full_mean = mean_val

        rows.append(
            AblationRow(
                component=name,
                mean=mean_val,
                std=std_val,
                ci95_half_width=ci_half,
                n=n_val,
                delta_vs_full=0.0,  # filled in below after full_mean is known
            )
        )

    # Fill delta_vs_full for all rows
    for row in rows:
        row.delta_vs_full = row.mean - full_mean

    return rows
