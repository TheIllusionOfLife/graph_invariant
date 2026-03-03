"""Frequency dominance analysis and epsilon sensitivity sweep.

Provides information-theoretic diagnostics explaining why frequency-based
link prediction performs well on small KGs with peaked edge-type
distributions, and an epsilon sensitivity sweep for the Harmony+Frequency
hybrid.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmony.metric.harmony import harmony_score
from harmony.types import KnowledgeGraph


def frequency_dominance_analysis(kg: KnowledgeGraph) -> dict[str, float]:
    """Compute frequency-dominance diagnostics for a KG.

    Returns a dict with:
      - edge_type_entropy: Shannon entropy of edge-type distribution (bits)
      - sparsity: 1 − (num_edges / (num_entities * (num_entities - 1)))
      - type_skewness: ratio of most-common to least-common edge-type count
      - p_max: probability of the most common edge type
      - theoretical_freq_advantage: p_max − 1/num_edge_types (advantage
        over uniform guessing)
    """
    if kg.num_edges == 0:
        return {
            "edge_type_entropy": 0.0,
            "sparsity": 1.0,
            "type_skewness": 0.0,
            "p_max": 0.0,
            "theoretical_freq_advantage": 0.0,
        }

    # Edge-type distribution
    edge_type_counts: Counter[str] = Counter()
    for edge in kg.edges:
        edge_type_counts[edge.edge_type.name] += 1

    total_edges = kg.num_edges
    probs = [c / total_edges for c in edge_type_counts.values()]

    # Shannon entropy (bits)
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    # Sparsity
    n = kg.num_entities
    max_edges = n * (n - 1) if n > 1 else 1
    sparsity = 1.0 - (total_edges / max_edges)

    # Skewness: ratio of max to min count
    counts = list(edge_type_counts.values())
    min_count = min(counts)
    max_count = max(counts)
    skewness = float(max_count / min_count) if min_count > 0 else float(max_count)

    # p_max
    p_max = max_count / total_edges

    # Number of distinct edge types used
    from harmony.types import EdgeType

    n_edge_types = len(EdgeType)
    theoretical_freq_advantage = p_max - (1.0 / n_edge_types)

    return {
        "edge_type_entropy": float(entropy),
        "sparsity": float(sparsity),
        "type_skewness": float(skewness),
        "p_max": float(p_max),
        "theoretical_freq_advantage": float(theoretical_freq_advantage),
    }


def epsilon_sensitivity_sweep(
    kg: KnowledgeGraph,
    epsilons: list[float] | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Sweep epsilon values and return Harmony scores for each.

    Parameters
    ----------
    kg:
        Knowledge graph to score.
    epsilons:
        List of epsilon values to sweep. Defaults to
        [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3].
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of dicts, each with keys: epsilon, harmony_score.
    """
    if epsilons is None:
        epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    results: list[dict[str, float]] = []
    for eps in epsilons:
        score = harmony_score(kg, epsilon=eps, seed=seed)
        results.append({"epsilon": eps, "harmony_score": score})

    return results
