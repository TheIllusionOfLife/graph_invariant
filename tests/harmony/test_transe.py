"""Tests for harmony.metric.transe — TDD-first.

Verifies:
  - TransE scoring: score(s, r, t) = -||e_s + r - e_t||
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


class TestTransE:
    def test_returns_float_in_unit_interval(self) -> None:
        from harmony.metric.transe import transe_hits_at_k

        kg = _make_sufficient_kg()
        score = transe_hits_at_k(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from harmony.metric.transe import transe_hits_at_k

        kg = _make_sufficient_kg()
        s1 = transe_hits_at_k(kg, seed=42)
        s2 = transe_hits_at_k(kg, seed=42)
        assert s1 == pytest.approx(s2)

    def test_empty_kg_returns_zero(self) -> None:
        from harmony.metric.transe import transe_hits_at_k

        kg = KnowledgeGraph(domain="empty")
        assert transe_hits_at_k(kg, seed=42) == 0.0

    def test_too_few_edges_returns_zero(self) -> None:
        from harmony.metric.transe import transe_hits_at_k

        kg = KnowledgeGraph(domain="tiny")
        for i in range(5):
            kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
        for i in range(3):
            kg.add_edge(
                TypedEdge(
                    source=f"e{i}",
                    target=f"e{i + 1}",
                    edge_type=EdgeType.DEPENDS_ON,
                )
            )
        assert transe_hits_at_k(kg, seed=42) == 0.0

    def test_score_function_uses_translation(self) -> None:
        """Verify the scoring is based on translation: -||h + r - t||."""
        from harmony.metric.transe import _TransE

        model = _TransE(entity_ids=["a", "b", "c"], dim=10, seed=42)
        # Set known embeddings for deterministic test
        model.E[0] = np.ones(10)  # a
        model.R[0] = np.ones(10) * 2  # relation 0
        model.E[1] = np.ones(10) * 3  # b = a + r exactly
        # Score(a, 0, b) should be 0 (perfect translation)
        score = model._score(0, 0, 1)
        assert abs(score) < 1e-6

    def test_training_improves_positive_triple_score(self) -> None:
        """Regression guard: training must perform gradient DESCENT.

        After training on a single positive triple (s, r, t), the score
        for that triple should increase (become less negative), proving
        the gradient update pushes in the correct direction.
        """
        from harmony.metric.transe import _TransE

        model = _TransE(entity_ids=["s", "t", "neg"], dim=10, seed=42)
        score_before = model._score(0, 0, 1)

        # Train on the single triple (s=0, r=0, t=1) for a few epochs
        model.train([(0, 0, 1)], n_epochs=20, lr=0.01, margin=1.0, n_neg=3, seed=42)
        score_after = model._score(0, 0, 1)

        # Score should improve (increase) after training — gradient descent
        assert score_after > score_before, (
            f"Training did not improve score: {score_before:.4f} → {score_after:.4f}. "
            "Gradient sign may be inverted."
        )
