"""Tests for harmony.metric.rotate — TDD-first.

Verifies:
  - RotatE scoring: score(s, r, t) = -||e_s ∘ r - e_t||
  - Returns float in [0, 1] for Hits@K
  - Deterministic under same seed
  - Returns 0.0 for empty/too-small KGs
"""

from __future__ import annotations

import numpy as np
import pytest

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


class TestRotatE:
    def test_returns_float_in_unit_interval(self) -> None:
        from harmony.metric.rotate import rotate_hits_at_k

        kg = _make_sufficient_kg()
        score = rotate_hits_at_k(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from harmony.metric.rotate import rotate_hits_at_k

        kg = _make_sufficient_kg()
        s1 = rotate_hits_at_k(kg, seed=42)
        s2 = rotate_hits_at_k(kg, seed=42)
        assert s1 == pytest.approx(s2)

    def test_empty_kg_returns_zero(self) -> None:
        from harmony.metric.rotate import rotate_hits_at_k

        kg = KnowledgeGraph(domain="empty")
        assert rotate_hits_at_k(kg, seed=42) == 0.0

    def test_matches_external_eval(self) -> None:
        """Standalone module should produce same result as external_eval."""
        from analysis.external_eval import evaluate_rotate
        from harmony.metric.rotate import rotate_hits_at_k

        kg = _make_sufficient_kg()
        standalone = rotate_hits_at_k(kg, seed=42)
        external = evaluate_rotate(kg, seed=42)
        assert abs(standalone - external) < 1e-10
