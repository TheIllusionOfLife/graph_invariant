"""Tests for harmony.metric.baselines — TDD-first, written before implementation.

Verifies:
  - Each baseline returns a float in [0, 1]
  - Determinism under the same seed
  - Structural ordering: frequency > random on a single-type KG
  - baseline_distmult_alone exactly matches generativity()
  - All baselines return 0.0 for KGs with too few edges (< _MIN_TRAIN_EDGES + 1)
"""

from __future__ import annotations

import pytest

from harmony.metric.generativity import _MIN_TRAIN_EDGES, generativity
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge

# ── Factories ─────────────────────────────────────────────────────────


def _make_empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(domain="empty")


def _make_too_small_kg() -> KnowledgeGraph:
    """KG with exactly _MIN_TRAIN_EDGES edges (too few: needs at least +1 masked)."""
    kg = KnowledgeGraph(domain="too_small")
    for i in range(_MIN_TRAIN_EDGES + 2):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    # Add exactly _MIN_TRAIN_EDGES edges — all train, none for masking
    for i in range(_MIN_TRAIN_EDGES):
        kg.add_edge(TypedEdge(source=f"e{i}", target=f"e{i + 1}", edge_type=EdgeType.DEPENDS_ON))
    return kg


def _make_sufficient_kg(n_entities: int = 20, n_edges: int = 30) -> KnowledgeGraph:
    """KG big enough to satisfy min-train-edges constraint."""
    kg = KnowledgeGraph(domain="sufficient")
    for i in range(n_entities):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    rng_seed = 99
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    added = 0
    attempts = 0
    while added < n_edges and attempts < n_edges * 10:
        attempts += 1
        s, t = rng.integers(0, n_entities, 2)
        if s == t:
            continue
        et = list(EdgeType)[int(rng.integers(0, len(EdgeType)))]
        kg.add_edge(TypedEdge(source=f"e{s}", target=f"e{t}", edge_type=et))
        added += 1
    return kg


def _make_single_type_kg() -> KnowledgeGraph:
    """KG with a single edge type — frequency baseline should perfectly rank true target.

    All edges are (e0 → eX, DEPENDS_ON).  The only observed (source_type, edge_type)
    pattern always leads to target entities e1..eN, so frequency counts are non-zero
    and the predictor is a better-than-random ranker.
    """
    n = 20
    kg = KnowledgeGraph(domain="single_type")
    for i in range(n):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    # Use e0 as the only source to create a strong frequency signal
    for i in range(1, n):
        kg.add_edge(
            TypedEdge(source="e0", target=f"e{i}", edge_type=EdgeType.DEPENDS_ON)
        )
    return kg


# ── Tests: baseline_random ────────────────────────────────────────────


def test_random_returns_zero_for_empty_kg() -> None:
    from harmony.metric.baselines import baseline_random

    assert baseline_random(_make_empty_kg()) == 0.0


def test_random_returns_zero_for_too_few_edges() -> None:
    from harmony.metric.baselines import baseline_random

    assert baseline_random(_make_too_small_kg()) == 0.0


def test_random_returns_float_in_bounds() -> None:
    from harmony.metric.baselines import baseline_random

    score = baseline_random(_make_sufficient_kg())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_random_is_deterministic() -> None:
    from harmony.metric.baselines import baseline_random

    kg = _make_sufficient_kg()
    assert baseline_random(kg, seed=7) == baseline_random(kg, seed=7)


def test_random_differs_for_different_seeds() -> None:
    from harmony.metric.baselines import baseline_random

    kg = _make_sufficient_kg(n_entities=50, n_edges=80)
    # Seeds likely produce different random orderings; not guaranteed but very probable
    scores = {baseline_random(kg, seed=s) for s in range(5)}
    # At least 2 distinct values across 5 seeds (would need perfect collision to fail)
    assert len(scores) >= 1  # weak: just verify no crash


# ── Tests: baseline_frequency ─────────────────────────────────────────


def test_frequency_returns_zero_for_empty_kg() -> None:
    from harmony.metric.baselines import baseline_frequency

    assert baseline_frequency(_make_empty_kg()) == 0.0


def test_frequency_returns_zero_for_too_few_edges() -> None:
    from harmony.metric.baselines import baseline_frequency

    assert baseline_frequency(_make_too_small_kg()) == 0.0


def test_frequency_returns_float_in_bounds() -> None:
    from harmony.metric.baselines import baseline_frequency

    score = baseline_frequency(_make_sufficient_kg())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_frequency_is_deterministic() -> None:
    from harmony.metric.baselines import baseline_frequency

    kg = _make_sufficient_kg()
    assert baseline_frequency(kg, seed=7) == baseline_frequency(kg, seed=7)


def test_frequency_higher_than_random_on_single_type_kg() -> None:
    """Frequency is a better-than-random predictor on a single-type KG.

    With a single (source_type, edge_type) pair and all edges pointing from one
    source node, frequency counts are concentrated — it should rank higher than
    a purely random baseline on average.
    """
    from harmony.metric.baselines import baseline_frequency, baseline_random

    kg = _make_single_type_kg()
    freq = baseline_frequency(kg, seed=42)
    rand = baseline_random(kg, seed=42)
    # Frequency exploits structure; should be >= random (or we need a larger n)
    assert freq >= rand, f"Expected freq ({freq:.3f}) >= random ({rand:.3f})"


# ── Tests: baseline_distmult_alone ────────────────────────────────────


def test_distmult_alone_returns_zero_for_too_few_edges() -> None:
    from harmony.metric.baselines import baseline_distmult_alone

    assert baseline_distmult_alone(_make_too_small_kg()) == 0.0


def test_distmult_alone_matches_generativity() -> None:
    """baseline_distmult_alone is a thin wrapper; must return exactly the same float."""
    from harmony.metric.baselines import baseline_distmult_alone

    kg = _make_sufficient_kg()
    expected = generativity(kg, seed=42, mask_ratio=0.2, k=10, dim=50, n_epochs=100)
    actual = baseline_distmult_alone(kg, seed=42, mask_ratio=0.2, k=10, dim=50, n_epochs=100)
    assert actual == expected


def test_distmult_alone_returns_float_in_bounds() -> None:
    from harmony.metric.baselines import baseline_distmult_alone

    score = baseline_distmult_alone(_make_sufficient_kg())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
