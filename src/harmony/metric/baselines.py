"""Baseline link-prediction scorers for Harmony metric calibration.

Three baselines, each using the same mask/evaluate protocol as generativity()
to allow apples-to-apples comparison:

  baseline_random           — Hits@K under uniform random target selection
  baseline_frequency        — Hits@K using (source_type, edge_type) frequency counts
  baseline_distmult_alone   — thin wrapper over generativity() (DistMult in isolation)

All return a float in [0, 1].  Return 0.0 when the KG has too few edges to
train/evaluate meaningfully (same threshold as generativity: _MIN_TRAIN_EDGES).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from harmony.metric.generativity import _MIN_TRAIN_EDGES
from harmony.types import EdgeType, KnowledgeGraph, TypedEdge

if TYPE_CHECKING:
    pass


def _split_edges(
    edges: list[TypedEdge],
    mask_ratio: float,
    seed: int,
) -> tuple[list[TypedEdge], list[TypedEdge]]:
    """Return (train_edges, test_edges) using the same protocol as generativity."""
    n_mask = max(1, int(len(edges) * mask_ratio))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    mask_idx: set[int] = set(int(i) for i in perm[:n_mask])
    train = [e for i, e in enumerate(edges) if i not in mask_idx]
    test = [e for i, e in enumerate(edges) if i in mask_idx]
    return train, test


def baseline_random(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
) -> float:
    """Hits@K under uniform random target selection.

    For each test edge: shuffle all entity IDs with the given seed (incremented
    per-edge to avoid identical shuffles), then check if the true target falls
    in the top-k positions.

    Returns 0.0 when the KG is empty or has too few edges.
    """
    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    n_train = len(edges) - max(1, int(len(edges) * mask_ratio))
    if n_train < _MIN_TRAIN_EDGES:
        return 0.0

    _, test_edges = _split_edges(edges, mask_ratio, seed)
    if not test_edges:
        return 0.0

    entity_ids = list(kg.entities.keys())
    n_entities = len(entity_ids)
    entity_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    effective_k = min(k, n_entities - 1)
    if effective_k <= 0:
        return 0.0

    rng = np.random.default_rng(seed)
    hits = 0
    for edge in test_edges:
        t_idx = entity_to_idx.get(edge.target)
        if t_idx is None:
            continue
        # Uniform random ranking: shuffle indices, check if true target in top-k
        shuffled = rng.permutation(n_entities)
        rank = int(np.where(shuffled == t_idx)[0][0])
        if rank < effective_k:
            hits += 1

    return hits / len(test_edges)


def baseline_frequency(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
) -> float:
    """Hits@K using frequency-based scoring.

    For each test triple (s, r, t_true): score target entities by how often
    they appear as a target for the same (source_entity_type, edge_type) pair
    in training edges.  Rank by descending frequency; ties broken randomly
    using rng.permutation on tied indices.

    Returns 0.0 when the KG is empty or has too few edges.
    """
    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    n_train = len(edges) - max(1, int(len(edges) * mask_ratio))
    if n_train < _MIN_TRAIN_EDGES:
        return 0.0

    train_edges, test_edges = _split_edges(edges, mask_ratio, seed)
    if not test_edges:
        return 0.0

    entity_ids = list(kg.entities.keys())
    n_entities = len(entity_ids)
    entity_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    effective_k = min(k, n_entities - 1)
    if effective_k <= 0:
        return 0.0

    # Build frequency table: (source_entity_type, edge_type) → Counter of target idx
    freq: dict[tuple[str, EdgeType], Counter[int]] = {}
    for e in train_edges:
        src_entity = kg.entities.get(e.source)
        tgt_idx = entity_to_idx.get(e.target)
        if src_entity is None or tgt_idx is None:
            continue
        key = (src_entity.entity_type, e.edge_type)
        if key not in freq:
            freq[key] = Counter()
        freq[key][tgt_idx] += 1

    rng = np.random.default_rng(seed)
    hits = 0
    for edge in test_edges:
        src_entity = kg.entities.get(edge.source)
        tgt_idx = entity_to_idx.get(edge.target)
        if src_entity is None or tgt_idx is None:
            continue

        key = (src_entity.entity_type, edge.edge_type)
        counts = freq.get(key, Counter())

        # Build score vector (frequency counts; unseen targets get 0)
        scores = np.array([float(counts.get(i, 0)) for i in range(n_entities)])

        # Break ties randomly: add tiny random noise proportional to a shuffle rank
        tie_break = rng.permutation(n_entities).astype(float) / n_entities
        scores = scores + tie_break * 1e-6

        top_k_indices = np.argsort(-scores)[:effective_k]
        if tgt_idx in top_k_indices:
            hits += 1

    return hits / len(test_edges)


def baseline_distmult_alone(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Hits@K using DistMult only — thin wrapper over generativity() for exact parity.

    Exposes the link-prediction component in isolation so the ablation table
    can show what generativity contributes vs. the composite metric.
    """
    from harmony.metric.generativity import generativity

    return generativity(kg, seed=seed, mask_ratio=mask_ratio, k=k, dim=dim, n_epochs=n_epochs)
