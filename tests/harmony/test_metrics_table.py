"""Tests for analysis.metrics_table — MRR + Hits@K comparison table.

TDD: written BEFORE implementation. Verifies:
  - compute_metrics_table returns DataFrame with expected shape and columns
  - Archive proposals are applied to KG before measurement
  - MRR returns 0.0 for trivial KGs (too few edges)
  - _mean_reciprocal_rank is in [0, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))

from harmony.map_elites import HarmonyMapElites, serialize_archive, try_insert
from harmony.proposals.types import Proposal, ProposalType
from harmony.state import HarmonySearchState, save_state
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_small_kg(n_entities: int = 5, n_edges: int = 4) -> KnowledgeGraph:
    """Build a minimal KG for unit tests (too few edges for DistMult)."""
    kg = KnowledgeGraph(domain="test")
    for i in range(n_entities):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    for i in range(n_edges):
        kg.add_edge(
            TypedEdge(
                source=f"e{i % n_entities}",
                target=f"e{(i + 1) % n_entities}",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
    return kg


def _make_medium_kg(n_entities: int = 20) -> KnowledgeGraph:
    """Build a KG with ≥15 edges (enough for DistMult training)."""
    kg = KnowledgeGraph(domain="test")
    for i in range(n_entities):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    edge_types = list(EdgeType)
    for i in range(15):
        kg.add_edge(
            TypedEdge(
                source=f"e{i % n_entities}",
                target=f"e{(i + 3) % n_entities}",
                edge_type=edge_types[i % len(edge_types)],
            )
        )
    return kg


def _make_add_edge_proposal(kg: KnowledgeGraph, pid: str) -> Proposal:
    """Return an ADD_EDGE proposal referencing entities that exist in kg."""
    entity_ids = list(kg.entities.keys())
    return Proposal(
        id=pid,
        proposal_type=ProposalType.ADD_EDGE,
        claim="This entity causally depends on another in the domain.",
        justification="Observed in multiple experimental studies consistently.",
        falsification_condition="If the prediction fails in controlled tests.",
        kg_domain="test",
        source_entity=entity_ids[0],
        target_entity=entity_ids[2],
        edge_type="EXPLAINS",
    )


def _checkpoint_dir(tmp_path: Path, domain: str) -> Path:
    """Create a checkpoint directory with a minimal checkpoint.json."""
    d = tmp_path / domain
    d.mkdir()
    state = HarmonySearchState(
        experiment_id=f"exp-{domain}",
        generation=5,
        islands={0: []},
        best_harmony_gain=0.1,
    )
    save_state(state, d / "checkpoint.json")
    return d


def _checkpoint_dir_with_archive(tmp_path: Path, domain: str, kg: KnowledgeGraph) -> Path:
    """Create a checkpoint directory with an archive containing one ADD_EDGE proposal."""
    d = tmp_path / domain
    d.mkdir()
    archive = HarmonyMapElites(num_bins=5)
    proposal = _make_add_edge_proposal(kg, "p001")
    try_insert(archive, proposal, fitness_signal=0.5, descriptor=(0.5, 0.5))
    state = HarmonySearchState(
        experiment_id=f"exp-{domain}",
        generation=5,
        islands={0: []},
        best_harmony_gain=0.1,
        archive=serialize_archive(archive),
    )
    save_state(state, d / "checkpoint.json")
    return d


class TestDomainBuilders:
    """Verify that _DOMAIN_BUILDERS covers all expected domains."""

    EXPECTED_DOMAINS = {
        "linear_algebra",
        "periodic_table",
        "astronomy",
        "physics",
        "materials",
        "wikidata_physics",
        "wikidata_materials",
    }

    def test_domain_builders_has_all_domains(self):
        from metrics_table import _DOMAIN_BUILDERS

        assert self.EXPECTED_DOMAINS == set(_DOMAIN_BUILDERS.keys())

    @pytest.mark.parametrize("domain", sorted(EXPECTED_DOMAINS))
    def test_domain_builder_importable(self, domain: str):
        """Each domain builder path resolves to a callable."""
        import importlib

        from metrics_table import _DOMAIN_BUILDERS

        module_path, func_name = _DOMAIN_BUILDERS[domain].rsplit(".", 1)
        module = importlib.import_module(module_path)
        assert callable(getattr(module, func_name))


class TestComputeMetricsTable:
    EXPECTED_COLUMNS = {
        "random_hits10",
        "freq_hits10",
        "distmult_hits10",
        "harmony_hits10",
        "mrr_random",
        "mrr_distmult",
        "mrr_harmony",
    }

    def test_compute_metrics_table_shape(self, tmp_path: Path):
        """DataFrame has one row per domain and all expected columns."""
        from metrics_table import compute_metrics_table

        kg_a = _make_small_kg()
        kg_b = _make_small_kg()

        dir_a = _checkpoint_dir(tmp_path, "domain_a")
        dir_b = _checkpoint_dir(tmp_path, "domain_b")

        df = compute_metrics_table(
            {"domain_a": dir_a, "domain_b": dir_b},
            kgs={"domain_a": kg_a, "domain_b": kg_b},
            seed=42,
        )

        assert df.shape[0] == 2
        assert self.EXPECTED_COLUMNS.issubset(set(df.columns))
        assert set(df.index) == {"domain_a", "domain_b"}

    def test_metrics_table_values_are_floats(self, tmp_path: Path):
        """All metric values are floats."""
        from metrics_table import compute_metrics_table

        kg = _make_small_kg()
        d = _checkpoint_dir(tmp_path, "dom")

        df = compute_metrics_table({"dom": d}, kgs={"dom": kg}, seed=42)

        for col in self.EXPECTED_COLUMNS:
            assert isinstance(df.loc["dom", col], float)

    def test_metrics_table_uses_archive_proposals(self, tmp_path: Path):
        """generativity is called with an augmented KG when archive has ADD_EDGE proposals."""
        from metrics_table import compute_metrics_table

        kg = _make_medium_kg()
        original_edge_count = kg.num_edges

        dir_d = _checkpoint_dir_with_archive(tmp_path, "domain_c", kg)

        captured_kgs: list[KnowledgeGraph] = []

        def _fake_generativity(kg_arg, seed=42, **kwargs):
            captured_kgs.append(kg_arg)
            return 0.5

        with patch("metrics_table.generativity", side_effect=_fake_generativity):
            compute_metrics_table({"domain_c": dir_d}, kgs={"domain_c": kg}, seed=42)

        # generativity must have been called with a KG that has more edges (augmented)
        assert len(captured_kgs) > 0
        augmented_kg = captured_kgs[0]
        assert augmented_kg.num_edges > original_edge_count

    def test_metrics_table_trivial_kg_returns_zeros(self, tmp_path: Path):
        """Small KG (< MIN_TRAIN_EDGES) → all metric values are 0.0."""
        from metrics_table import compute_metrics_table

        # 3 entities, 2 edges — well below _MIN_TRAIN_EDGES=10
        kg = _make_small_kg(n_entities=3, n_edges=2)
        d = _checkpoint_dir(tmp_path, "tiny")

        df = compute_metrics_table({"tiny": d}, kgs={"tiny": kg}, seed=42)

        for col in self.EXPECTED_COLUMNS:
            assert df.loc["tiny", col] == pytest.approx(0.0), f"{col} should be 0.0 for trivial KG"


class TestMeanReciprocalRank:
    def test_mrr_returns_zero_for_trivial_kg(self):
        """KG with < MIN_TRAIN_EDGES → MRR = 0.0."""
        from metrics_table import _mean_reciprocal_rank

        kg = _make_small_kg(n_entities=3, n_edges=2)
        result = _mean_reciprocal_rank(kg, seed=42)
        assert result == pytest.approx(0.0)

    def test_mrr_returns_zero_for_empty_kg(self):
        """Empty KG → MRR = 0.0."""
        from metrics_table import _mean_reciprocal_rank

        kg = KnowledgeGraph(domain="empty")
        result = _mean_reciprocal_rank(kg, seed=42)
        assert result == pytest.approx(0.0)

    def test_mrr_in_unit_interval(self):
        """MRR ∈ [0, 1] for any valid KG."""
        from metrics_table import _mean_reciprocal_rank

        kg = _make_medium_kg(n_entities=20)
        result = _mean_reciprocal_rank(kg, seed=42)
        assert 0.0 <= result <= 1.0

    def test_mrr_deterministic(self):
        """Same seed → same result."""
        from metrics_table import _mean_reciprocal_rank

        kg = _make_medium_kg(n_entities=20)
        r1 = _mean_reciprocal_rank(kg, seed=7)
        r2 = _mean_reciprocal_rank(kg, seed=7)
        assert r1 == pytest.approx(r2)


class TestApplyProposalsToKg:
    def test_apply_proposals_adds_valid_edge(self):
        """ADD_EDGE proposal with existing entities adds an edge to the copy."""
        from metrics_table import _apply_proposals_to_kg

        kg = _make_medium_kg(n_entities=10)
        original_count = kg.num_edges

        entity_ids = list(kg.entities.keys())
        proposal = Proposal(
            id="p001",
            proposal_type=ProposalType.ADD_EDGE,
            claim="This entity causally depends on another in the domain.",
            justification="Observed in multiple experimental studies consistently.",
            falsification_condition="If the prediction fails in controlled tests.",
            kg_domain="test",
            source_entity=entity_ids[0],
            target_entity=entity_ids[5],
            edge_type="EXPLAINS",
        )

        augmented = _apply_proposals_to_kg(kg, [proposal])

        assert augmented.num_edges == original_count + 1
        # Original KG should be unmodified (deep copy)
        assert kg.num_edges == original_count

    def test_apply_proposals_skips_unknown_entities(self):
        """Proposals referencing non-existent entities are silently skipped."""
        from metrics_table import _apply_proposals_to_kg

        kg = _make_medium_kg(n_entities=5)
        original_count = kg.num_edges

        proposal = Proposal(
            id="p002",
            proposal_type=ProposalType.ADD_EDGE,
            claim="This entity causally depends on another in the domain.",
            justification="Observed in multiple experimental studies consistently.",
            falsification_condition="If the prediction fails in controlled tests.",
            kg_domain="test",
            source_entity="nonexistent_entity",
            target_entity="also_nonexistent",
            edge_type="EXPLAINS",
        )

        augmented = _apply_proposals_to_kg(kg, [proposal])
        assert augmented.num_edges == original_count

    def test_apply_proposals_does_not_mutate_original(self):
        """The original KG is never mutated — only the returned copy is changed."""
        from metrics_table import _apply_proposals_to_kg

        kg = _make_medium_kg(n_entities=10)
        entity_ids = list(kg.entities.keys())
        original_count = kg.num_edges

        proposal = Proposal(
            id="p003",
            proposal_type=ProposalType.ADD_EDGE,
            claim="This entity causally depends on another in the domain.",
            justification="Observed in multiple experimental studies consistently.",
            falsification_condition="If the prediction fails in controlled tests.",
            kg_domain="test",
            source_entity=entity_ids[0],
            target_entity=entity_ids[4],
            edge_type="GENERALIZES",
        )

        _apply_proposals_to_kg(kg, [proposal])
        assert kg.num_edges == original_count
