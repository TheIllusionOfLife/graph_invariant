"""Tests for analysis.harmony_downstream_analysis.

TDD: written BEFORE implementation. Verifies:
  - component_correlation_analysis returns expected structure
  - failure_mode_analysis returns expected categories
  - regime_characterization returns scatter data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_test_kg() -> KnowledgeGraph:
    """Small KG for unit tests."""
    kg = KnowledgeGraph(domain="test")
    for i in range(10):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    for i in range(10):
        kg.add_edge(
            TypedEdge(
                source=f"e{i}",
                target=f"e{(i + 1) % 10}",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
    return kg


def _make_proposal_scores() -> list[dict[str, float]]:
    """Fake per-proposal component deltas + Hits@10 delta."""
    return [
        {
            "delta_compress": 0.1,
            "delta_cohere": 0.05,
            "delta_symm": 0.0,
            "delta_gener": 0.2,
            "delta_hits10": 0.15,
        },
        {
            "delta_compress": -0.05,
            "delta_cohere": 0.1,
            "delta_symm": 0.02,
            "delta_gener": -0.1,
            "delta_hits10": -0.05,
        },
        {
            "delta_compress": 0.2,
            "delta_cohere": -0.03,
            "delta_symm": 0.1,
            "delta_gener": 0.3,
            "delta_hits10": 0.25,
        },
        {
            "delta_compress": 0.0,
            "delta_cohere": 0.0,
            "delta_symm": 0.0,
            "delta_gener": 0.0,
            "delta_hits10": 0.0,
        },
        {
            "delta_compress": 0.15,
            "delta_cohere": 0.08,
            "delta_symm": -0.01,
            "delta_gener": 0.1,
            "delta_hits10": 0.1,
        },
    ]


class TestComponentCorrelationAnalysis:
    def test_returns_expected_keys(self):
        from harmony_downstream_analysis import component_correlation_analysis

        scores = _make_proposal_scores()
        result = component_correlation_analysis(scores)
        assert "correlations" in result
        assert "p_values" in result

    def test_correlations_in_range(self):
        from harmony_downstream_analysis import component_correlation_analysis

        result = component_correlation_analysis(_make_proposal_scores())
        for comp, rho in result["correlations"].items():
            assert -1.0 <= rho <= 1.0, f"{comp} correlation {rho} out of range"

    def test_all_components_present(self):
        from harmony_downstream_analysis import component_correlation_analysis

        result = component_correlation_analysis(_make_proposal_scores())
        expected_components = {"delta_compress", "delta_cohere", "delta_symm", "delta_gener"}
        assert set(result["correlations"].keys()) == expected_components


class TestFailureModeAnalysis:
    def test_returns_expected_keys(self):
        from harmony_downstream_analysis import failure_mode_analysis

        proposals = [
            {
                "proposal_type": "ADD_EDGE",
                "edge_type": "DEPENDS_ON",
                "valid": True,
                "hits10_delta": 0.1,
            },
            {
                "proposal_type": "ADD_EDGE",
                "edge_type": "CONTRADICTS",
                "valid": True,
                "hits10_delta": -0.05,
            },
            {"proposal_type": "ADD_ENTITY", "edge_type": None, "valid": False, "hits10_delta": 0.0},
        ]
        result = failure_mode_analysis(proposals)
        assert "type_distribution" in result
        assert "edge_type_distribution" in result
        assert "classification" in result

    def test_classification_categories(self):
        from harmony_downstream_analysis import failure_mode_analysis

        proposals = [
            {
                "proposal_type": "ADD_EDGE",
                "edge_type": "DEPENDS_ON",
                "valid": True,
                "hits10_delta": 0.1,
            },
            {
                "proposal_type": "ADD_EDGE",
                "edge_type": "CONTRADICTS",
                "valid": True,
                "hits10_delta": 0.0,
            },
            {
                "proposal_type": "ADD_EDGE",
                "edge_type": "DERIVES",
                "valid": True,
                "hits10_delta": -0.05,
            },
        ]
        result = failure_mode_analysis(proposals)
        expected_categories = {"valid_helpful", "valid_neutral", "valid_harmful"}
        assert expected_categories.issubset(set(result["classification"].keys()))


class TestRegimeCharacterization:
    def test_returns_expected_structure(self):
        from harmony_downstream_analysis import regime_characterization

        domain_results = [
            {"domain": "linear_algebra", "density": 0.3, "entropy": 1.5, "harmony_freq_gap": 0.1},
            {"domain": "periodic_table", "density": 0.1, "entropy": 2.0, "harmony_freq_gap": -0.05},
            {"domain": "astronomy", "density": 0.05, "entropy": 2.5, "harmony_freq_gap": -0.15},
        ]
        result = regime_characterization(domain_results)
        assert "density_vs_gap" in result
        assert "entropy_vs_gap" in result

    def test_scatter_data_shape(self):
        from harmony_downstream_analysis import regime_characterization

        domain_results = [
            {"domain": "d1", "density": 0.3, "entropy": 1.5, "harmony_freq_gap": 0.1},
            {"domain": "d2", "density": 0.1, "entropy": 2.0, "harmony_freq_gap": -0.05},
        ]
        result = regime_characterization(domain_results)
        assert len(result["density_vs_gap"]) == 2
        assert len(result["entropy_vs_gap"]) == 2
        for point in result["density_vs_gap"]:
            assert "x" in point and "y" in point and "domain" in point
