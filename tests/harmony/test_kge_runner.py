"""Tests for harmony.metric.kge_baselines unified runner — TDD-first."""

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


class TestRunAllKgeBaselines:
    def test_returns_all_six_models(self) -> None:
        from harmony.metric.kge_baselines import run_all_kge_baselines

        kg = _make_sufficient_kg()
        results = run_all_kge_baselines(kg, seed=42)
        expected_keys = {"random", "frequency", "distmult", "transe", "rotate", "complex"}
        assert set(results.keys()) == expected_keys

    def test_all_values_in_unit_interval(self) -> None:
        from harmony.metric.kge_baselines import run_all_kge_baselines

        kg = _make_sufficient_kg()
        results = run_all_kge_baselines(kg, seed=42)
        for model, score in results.items():
            assert isinstance(score, float), f"{model} not float"
            assert 0.0 <= score <= 1.0, f"{model} out of range: {score}"

    def test_deterministic(self) -> None:
        from harmony.metric.kge_baselines import run_all_kge_baselines

        kg = _make_sufficient_kg()
        r1 = run_all_kge_baselines(kg, seed=42)
        r2 = run_all_kge_baselines(kg, seed=42)
        for model in r1:
            assert r1[model] == pytest.approx(r2[model])

    def test_empty_kg_returns_all_zeros(self) -> None:
        from harmony.metric.kge_baselines import run_all_kge_baselines

        kg = KnowledgeGraph(domain="empty")
        results = run_all_kge_baselines(kg, seed=42)
        for model, score in results.items():
            assert score == 0.0, f"{model} should be 0.0 for empty KG"


class TestConceptualBaselines:
    def test_llm_only_config_structure(self) -> None:
        from harmony.metric.kge_baselines import LLM_ONLY_CONFIG

        assert LLM_ONLY_CONFIG["accept_all_valid"] is True
        assert LLM_ONLY_CONFIG["alpha"] == 0.0

    def test_no_qd_config_structure(self) -> None:
        from harmony.metric.kge_baselines import NO_QD_CONFIG

        assert NO_QD_CONFIG["num_bins"] == 1
        assert NO_QD_CONFIG["greedy"] is True
