"""Tests for analysis.wikidata_ablation and analysis.reproducibility_table.

TDD: written BEFORE implementation. Verifies:
  - run_wikidata_ablation returns DataFrame with expected shape
  - generate_reproducibility_table returns DataFrame with all domains
  - All expected columns present
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_test_kg(domain: str = "test", n_entities: int = 15) -> KnowledgeGraph:
    """Build a test KG with enough edges for ablation."""
    kg = KnowledgeGraph(domain=domain)
    for i in range(n_entities):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    edge_types = list(EdgeType)
    for i in range(n_entities):
        kg.add_edge(
            TypedEdge(
                source=f"e{i}",
                target=f"e{(i + 1) % n_entities}",
                edge_type=edge_types[i % len(edge_types)],
            )
        )
    return kg


class TestWikidataAblation:
    def test_returns_dataframe(self):
        from wikidata_ablation import run_wikidata_ablation

        kgs = {"domain_a": _make_test_kg("domain_a"), "domain_b": _make_test_kg("domain_b")}
        df = run_wikidata_ablation(kgs, n_bootstrap=10, seed=42)
        assert df.shape[0] > 0

    def test_has_expected_columns(self):
        from wikidata_ablation import run_wikidata_ablation

        kgs = {"domain_a": _make_test_kg("domain_a")}
        df = run_wikidata_ablation(kgs, n_bootstrap=10, seed=42)
        expected = {"domain", "component", "mean", "std", "ci95_low", "ci95_high"}
        assert expected.issubset(set(df.columns))

    def test_components_in_expected_set(self):
        from wikidata_ablation import run_wikidata_ablation

        kgs = {"domain_a": _make_test_kg("domain_a")}
        df = run_wikidata_ablation(kgs, n_bootstrap=10, seed=42)
        expected_components = {"compressibility", "coherence", "symmetry", "generativity"}
        actual_components = set(df["component"].unique())
        assert expected_components == actual_components

    def test_means_in_bounds(self):
        from wikidata_ablation import run_wikidata_ablation

        kgs = {"test": _make_test_kg("test")}
        df = run_wikidata_ablation(kgs, n_bootstrap=10, seed=42)
        for _, row in df.iterrows():
            assert 0.0 <= row["mean"] <= 1.0


class TestReproducibilityTable:
    ALL_DOMAINS = {
        "linear_algebra",
        "periodic_table",
        "astronomy",
        "physics",
        "materials",
        "wikidata_physics",
        "wikidata_materials",
    }

    EXPECTED_COLUMNS = {
        "domain",
        "entities",
        "edges",
        "entity_types",
        "edge_types",
        "source",
    }

    def test_returns_dataframe(self):
        from reproducibility_table import generate_reproducibility_table

        df = generate_reproducibility_table()
        assert df.shape[0] > 0

    def test_has_all_domains(self):
        from reproducibility_table import generate_reproducibility_table

        df = generate_reproducibility_table()
        assert self.ALL_DOMAINS.issubset(set(df["domain"]))

    def test_has_expected_columns(self):
        from reproducibility_table import generate_reproducibility_table

        df = generate_reproducibility_table()
        assert self.EXPECTED_COLUMNS.issubset(set(df.columns))

    def test_entity_counts_positive(self):
        from reproducibility_table import generate_reproducibility_table

        df = generate_reproducibility_table()
        for _, row in df.iterrows():
            assert row["entities"] > 0
            assert row["edges"] > 0
