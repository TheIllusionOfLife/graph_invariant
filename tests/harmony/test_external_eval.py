"""Tests for analysis.external_eval — TDD-first, written before implementation.

Verifies:
  - RotatE evaluation returns Hits@K ∈ [0, 1]
  - ComplEx evaluation returns Hits@K ∈ [0, 1]
  - Determinism under the same seed
  - Both return 0.0 for KGs with too few edges
  - evaluate_external() returns results for both models
  - NO_GENERATIVITY ablation variant works correctly
"""

from __future__ import annotations

import numpy as np

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge

# ── Factories ─────────────────────────────────────────────────────────


def _make_empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(domain="empty")


def _make_sufficient_kg(n_entities: int = 20, n_edges: int = 30) -> KnowledgeGraph:
    """KG big enough to satisfy min-train-edges constraint."""
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


# ── RotatE evaluation ────────────────────────────────────────────────


class TestRotatEEval:
    def test_returns_float_in_unit_interval(self) -> None:
        from analysis.external_eval import evaluate_rotate

        kg = _make_sufficient_kg()
        score = evaluate_rotate(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from analysis.external_eval import evaluate_rotate

        kg = _make_sufficient_kg()
        s1 = evaluate_rotate(kg, seed=42)
        s2 = evaluate_rotate(kg, seed=42)
        assert s1 == s2

    def test_empty_kg_returns_zero(self) -> None:
        from analysis.external_eval import evaluate_rotate

        kg = _make_empty_kg()
        assert evaluate_rotate(kg, seed=42) == 0.0

    def test_too_few_edges_returns_zero(self) -> None:
        from analysis.external_eval import evaluate_rotate

        kg = KnowledgeGraph(domain="tiny")
        for i in range(5):
            kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
        for i in range(3):
            kg.add_edge(
                TypedEdge(source=f"e{i}", target=f"e{i + 1}", edge_type=EdgeType.DEPENDS_ON)
            )
        assert evaluate_rotate(kg, seed=42) == 0.0


# ── ComplEx evaluation ───────────────────────────────────────────────


class TestComplExEval:
    def test_returns_float_in_unit_interval(self) -> None:
        from analysis.external_eval import evaluate_complex

        kg = _make_sufficient_kg()
        score = evaluate_complex(kg, seed=42)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        from analysis.external_eval import evaluate_complex

        kg = _make_sufficient_kg()
        s1 = evaluate_complex(kg, seed=42)
        s2 = evaluate_complex(kg, seed=42)
        assert s1 == s2

    def test_empty_kg_returns_zero(self) -> None:
        from analysis.external_eval import evaluate_complex

        kg = _make_empty_kg()
        assert evaluate_complex(kg, seed=42) == 0.0

    def test_too_few_edges_returns_zero(self) -> None:
        from analysis.external_eval import evaluate_complex

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
        assert evaluate_complex(kg, seed=42) == 0.0


# ── Unified evaluation ───────────────────────────────────────────────


class TestEvaluateExternal:
    def test_returns_all_models(self) -> None:
        from analysis.external_eval import evaluate_external

        kg = _make_sufficient_kg()
        results = evaluate_external(kg, seed=42)
        assert "rotate" in results
        assert "complex" in results
        assert "distmult" in results

    def test_values_are_floats(self) -> None:
        from analysis.external_eval import evaluate_external

        kg = _make_sufficient_kg()
        results = evaluate_external(kg, seed=42)
        for model_name, score in results.items():
            assert isinstance(score, float), f"{model_name} score is not float"
            assert 0.0 <= score <= 1.0, f"{model_name} score out of range"

    def test_empty_kg_returns_all_zeros(self) -> None:
        from analysis.external_eval import evaluate_external

        kg = _make_empty_kg()
        results = evaluate_external(kg, seed=42)
        for model_name, score in results.items():
            assert score == 0.0, f"{model_name} should be 0.0 for empty KG"

    def test_models_produce_different_scores(self) -> None:
        """Sanity check: at least two models differ on a sufficient KG."""
        from analysis.external_eval import evaluate_external

        kg = _make_sufficient_kg()
        results = evaluate_external(kg, seed=42)
        scores = list(results.values())
        # At least two models should differ (they use different scoring functions)
        assert len(set(scores)) >= 2, "All models returned identical scores"


# ── NO_GENERATIVITY ablation ─────────────────────────────────────────


class TestNoGenerativityAblation:
    def test_no_generativity_variant_exists(self) -> None:
        from harmony.metric.ablation import run_ablation

        kg = _make_sufficient_kg()
        rows = run_ablation(kg, n_bootstrap=5, seed=42)
        component_names = [r.component for r in rows]
        assert "w/o_gen" in component_names

    def test_no_generativity_score_differs_from_full(self) -> None:
        from harmony.metric.ablation import run_ablation

        kg = _make_sufficient_kg()
        rows = run_ablation(kg, n_bootstrap=5, seed=42)
        full_row = next(r for r in rows if r.component == "full")
        no_gen_row = next(r for r in rows if r.component == "w/o_gen")
        # With generativity zeroed, score should differ from full
        # (generativity contributes non-trivially to the composite score)
        assert abs(no_gen_row.mean - full_row.mean) > 1e-6
