"""Tests for analysis.frequency_analysis — frequency dominance diagnostics.

TDD: written BEFORE implementation. Verifies:
  - frequency_dominance_analysis returns expected diagnostic dict
  - Edge cases: empty KG, single-edge-type, uniform edge types
  - epsilon_sensitivity_sweep returns expected structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(domain="empty")


def _make_peaked_kg() -> KnowledgeGraph:
    """KG where one edge type dominates (90% DEPENDS_ON)."""
    kg = KnowledgeGraph(domain="peaked")
    for i in range(10):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    # 9 DEPENDS_ON, 1 DERIVES
    for i in range(9):
        kg.add_edge(
            TypedEdge(source=f"e{i}", target=f"e{(i + 1) % 10}", edge_type=EdgeType.DEPENDS_ON)
        )
    kg.add_edge(TypedEdge(source="e0", target="e5", edge_type=EdgeType.DERIVES))
    return kg


def _make_uniform_kg() -> KnowledgeGraph:
    """KG with roughly equal distribution of edge types."""
    kg = KnowledgeGraph(domain="uniform")
    for i in range(14):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    edge_types = list(EdgeType)
    for i in range(14):
        kg.add_edge(
            TypedEdge(
                source=f"e{i}",
                target=f"e{(i + 1) % 14}",
                edge_type=edge_types[i % len(edge_types)],
            )
        )
    return kg


class TestFrequencyDominanceAnalysis:
    def test_returns_expected_keys(self):
        from frequency_analysis import frequency_dominance_analysis

        kg = _make_peaked_kg()
        result = frequency_dominance_analysis(kg)
        assert "edge_type_entropy" in result
        assert "sparsity" in result
        assert "type_skewness" in result
        assert "p_max" in result
        assert "theoretical_freq_advantage" in result

    def test_peaked_kg_low_entropy(self):
        from frequency_analysis import frequency_dominance_analysis

        result = frequency_dominance_analysis(_make_peaked_kg())
        # Peaked distribution → low entropy
        assert result["edge_type_entropy"] < 1.0
        # High p_max (dominant edge type)
        assert result["p_max"] > 0.8

    def test_uniform_kg_higher_entropy(self):
        from frequency_analysis import frequency_dominance_analysis

        peaked = frequency_dominance_analysis(_make_peaked_kg())
        uniform = frequency_dominance_analysis(_make_uniform_kg())
        assert uniform["edge_type_entropy"] > peaked["edge_type_entropy"]

    def test_empty_kg(self):
        from frequency_analysis import frequency_dominance_analysis

        result = frequency_dominance_analysis(_make_empty_kg())
        assert result["edge_type_entropy"] == pytest.approx(0.0)
        assert result["p_max"] == pytest.approx(0.0)

    def test_values_are_floats(self):
        from frequency_analysis import frequency_dominance_analysis

        result = frequency_dominance_analysis(_make_peaked_kg())
        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float, got {type(val)}"


class TestEpsilonSensitivity:
    def test_sweep_returns_list_of_dicts(self):
        from frequency_analysis import epsilon_sensitivity_sweep

        kg = _make_peaked_kg()
        results = epsilon_sensitivity_sweep(kg, epsilons=[0.0, 0.1, 0.2])
        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert "epsilon" in r
            assert "harmony_score" in r

    def test_sweep_epsilon_values_match_input(self):
        from frequency_analysis import epsilon_sensitivity_sweep

        epsilons = [0.0, 0.15, 0.3]
        results = epsilon_sensitivity_sweep(_make_peaked_kg(), epsilons=epsilons)
        returned_epsilons = [r["epsilon"] for r in results]
        assert returned_epsilons == epsilons

    def test_sweep_scores_in_unit_interval(self):
        from frequency_analysis import epsilon_sensitivity_sweep

        results = epsilon_sensitivity_sweep(_make_peaked_kg(), epsilons=[0.0, 0.1, 0.2, 0.3])
        for r in results:
            assert 0.0 <= r["harmony_score"] <= 1.0
