"""Tests for KGE baselines (TransE/RotatE/ComplEx) in baselines.py — TDD-first.

Verifies:
  - baseline_transe, baseline_rotate, baseline_complex return float in [0, 1]
  - All deterministic under same seed
  - All return 0.0 for empty KGs
  - baseline_transe matches transe_hits_at_k from standalone module
"""

from __future__ import annotations

import numpy as np

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_sufficient_kg(n_entities: int = 20, n_edges: int = 30) -> KnowledgeGraph:
    kg = KnowledgeGraph(domain="sufficient")
    for i in range(n_entities):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    rng = np.random.default_rng(99)
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


class TestBaselineTransE:
    def test_returns_float_in_unit_interval(self) -> None:
        from harmony.metric.baselines import baseline_transe

        kg = _make_sufficient_kg()
        score = baseline_transe(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from harmony.metric.baselines import baseline_transe

        kg = _make_sufficient_kg()
        s1 = baseline_transe(kg, seed=42)
        s2 = baseline_transe(kg, seed=42)
        assert s1 == s2

    def test_empty_kg_returns_zero(self) -> None:
        from harmony.metric.baselines import baseline_transe

        kg = KnowledgeGraph(domain="empty")
        assert baseline_transe(kg, seed=42) == 0.0

    def test_matches_standalone(self) -> None:
        from harmony.metric.baselines import baseline_transe
        from harmony.metric.transe import transe_hits_at_k

        kg = _make_sufficient_kg()
        assert baseline_transe(kg, seed=42) == transe_hits_at_k(kg, seed=42)


class TestBaselineRotatE:
    def test_returns_float_in_unit_interval(self) -> None:
        from harmony.metric.baselines import baseline_rotate

        kg = _make_sufficient_kg()
        score = baseline_rotate(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from harmony.metric.baselines import baseline_rotate

        kg = _make_sufficient_kg()
        s1 = baseline_rotate(kg, seed=42)
        s2 = baseline_rotate(kg, seed=42)
        assert s1 == s2

    def test_empty_kg_returns_zero(self) -> None:
        from harmony.metric.baselines import baseline_rotate

        kg = KnowledgeGraph(domain="empty")
        assert baseline_rotate(kg, seed=42) == 0.0


class TestBaselineComplEx:
    def test_returns_float_in_unit_interval(self) -> None:
        from harmony.metric.baselines import baseline_complex

        kg = _make_sufficient_kg()
        score = baseline_complex(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from harmony.metric.baselines import baseline_complex

        kg = _make_sufficient_kg()
        s1 = baseline_complex(kg, seed=42)
        s2 = baseline_complex(kg, seed=42)
        assert s1 == s2

    def test_empty_kg_returns_zero(self) -> None:
        from harmony.metric.baselines import baseline_complex

        kg = KnowledgeGraph(domain="empty")
        assert baseline_complex(kg, seed=42) == 0.0
