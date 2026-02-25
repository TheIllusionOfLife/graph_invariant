"""Tests for harmony.proposals.pipeline — end-to-end scoring, archiving, valid_rate.

TDD: these tests are written BEFORE implementation. They verify:
  - Every proposal gets a ProposalResult entry
  - Invalid proposals have harmony_gain=None
  - Valid proposals have a float harmony_gain
  - valid_rate is computed correctly as fraction validated
  - Pipeline is deterministic given the same seed
  - Archive is populated with valid proposals
"""

from __future__ import annotations

import pytest

from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.proposals.pipeline import run_pipeline
from harmony.proposals.types import Proposal, ProposalType


def _make_valid_add_entity_proposal(pid: str = "p1") -> Proposal:
    return Proposal(
        id=pid,
        proposal_type=ProposalType.ADD_ENTITY,
        claim="A new mathematical concept can be formalized in this domain.",
        justification="It is observed in multiple proofs and theorems consistently.",
        falsification_condition="If the concept reduces to an existing one without loss.",
        kg_domain="linear_algebra",
        entity_id=f"novel_entity_{pid}",
        entity_type="concept",
    )


def _make_invalid_proposal(pid: str = "pinv") -> Proposal:
    return Proposal(
        id=pid,
        proposal_type=ProposalType.ADD_EDGE,
        claim="hi",  # too short — fails validation
        justification="no",
        falsification_condition="nope",
        kg_domain="la",
        source_entity="a",
        target_entity="b",
        edge_type="DEPENDS_ON",
    )


@pytest.fixture
def kg():
    return build_linear_algebra_kg()


class TestPipelineResults:
    def test_all_proposals_have_result(self, kg):
        proposals = [_make_valid_add_entity_proposal("p1"), _make_invalid_proposal("p2")]
        result = run_pipeline(kg, proposals, seed=42)
        assert len(result.results) == 2

    def test_invalid_proposals_have_null_gain(self, kg):
        proposals = [_make_invalid_proposal("pinv")]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.results[0].harmony_gain is None

    def test_invalid_proposals_have_false_inserted(self, kg):
        proposals = [_make_invalid_proposal("pinv")]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.results[0].inserted_to_archive is False

    def test_valid_proposal_has_numeric_gain(self, kg):
        proposals = [_make_valid_add_entity_proposal("p1")]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.results[0].harmony_gain is not None
        assert isinstance(result.results[0].harmony_gain, float)

    def test_results_preserve_proposal_order(self, kg):
        proposals = [
            _make_valid_add_entity_proposal("p1"),
            _make_invalid_proposal("pinv"),
            _make_valid_add_entity_proposal("p3"),
        ]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.results[0].proposal.id == "p1"
        assert result.results[1].proposal.id == "pinv"
        assert result.results[2].proposal.id == "p3"


class TestValidRate:
    def test_valid_rate_correct(self, kg):
        proposals = [
            _make_valid_add_entity_proposal("p1"),
            _make_valid_add_entity_proposal("p2"),
            _make_invalid_proposal("pinv"),
        ]
        result = run_pipeline(kg, proposals, seed=42)
        assert abs(result.valid_rate - 2 / 3) < 1e-6

    def test_valid_rate_all_valid(self, kg):
        proposals = [_make_valid_add_entity_proposal(f"p{i}") for i in range(3)]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.valid_rate == pytest.approx(1.0)

    def test_valid_rate_all_invalid(self, kg):
        proposals = [_make_invalid_proposal(f"pinv{i}") for i in range(3)]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.valid_rate == pytest.approx(0.0)

    def test_valid_rate_empty_proposals(self, kg):
        result = run_pipeline(kg, [], seed=42)
        assert result.valid_rate == pytest.approx(0.0)
        assert result.results == []


class TestDeterminism:
    def test_pipeline_deterministic(self, kg):
        proposals = [_make_valid_add_entity_proposal("p1")]
        result1 = run_pipeline(kg, proposals, seed=42)
        result2 = run_pipeline(kg, proposals, seed=42)
        assert result1.results[0].harmony_gain == result2.results[0].harmony_gain

    def test_pipeline_same_seed_same_valid_rate(self, kg):
        proposals = [_make_valid_add_entity_proposal("p1"), _make_invalid_proposal("p2")]
        r1 = run_pipeline(kg, proposals, seed=42)
        r2 = run_pipeline(kg, proposals, seed=42)
        assert r1.valid_rate == r2.valid_rate


class TestMutationPaths:
    """Coverage for ADD_EDGE, REMOVE_EDGE, REMOVE_ENTITY mutation branches."""

    def test_add_edge_produces_numeric_gain(self, kg):
        # "rank" and "field" both exist; GENERALIZES not yet between them
        p = Proposal(
            id="add-edge",
            proposal_type=ProposalType.ADD_EDGE,
            claim="Rank generalizes over the underlying field structure in linear algebra.",
            justification="The rank concept depends on the scalar field for its definition.",
            falsification_condition="If rank can be defined without reference to the field.",
            kg_domain="linear_algebra",
            source_entity="rank",
            target_entity="field",
            edge_type="GENERALIZES",
        )
        result = run_pipeline(kg, [p], seed=42)
        assert result.results[0].harmony_gain is not None
        assert isinstance(result.results[0].harmony_gain, float)

    def test_remove_edge_produces_numeric_gain(self, kg):
        # First edge in the KG: vector_space → field (DEPENDS_ON)
        p = Proposal(
            id="remove-edge",
            proposal_type=ProposalType.REMOVE_EDGE,
            claim="The DEPENDS_ON edge from vector_space to field is redundant here.",
            justification="Other edges already capture the field dependency transitively.",
            falsification_condition="If removing it breaks any structural property of the KG.",
            kg_domain="linear_algebra",
            source_entity="vector_space",
            target_entity="field",
            edge_type="DEPENDS_ON",
        )
        result = run_pipeline(kg, [p], seed=42)
        assert result.results[0].harmony_gain is not None
        assert isinstance(result.results[0].harmony_gain, float)

    def test_remove_entity_produces_numeric_gain(self, kg):
        p = Proposal(
            id="remove-entity",
            proposal_type=ProposalType.REMOVE_ENTITY,
            claim="The quotient_space entity is redundant in this minimal KG formulation.",
            justification="It duplicates concepts already encoded by direct_sum relationships.",
            falsification_condition="If downstream theorems explicitly reference quotient_space.",
            kg_domain="linear_algebra",
            entity_id="quotient_space",
        )
        result = run_pipeline(kg, [p], seed=42)
        assert result.results[0].harmony_gain is not None
        assert isinstance(result.results[0].harmony_gain, float)


class TestArchive:
    def test_archive_populated_by_valid_proposals(self, kg):
        proposals = [_make_valid_add_entity_proposal(f"p{i}") for i in range(3)]
        result = run_pipeline(kg, proposals, seed=42)
        inserted = sum(1 for r in result.results if r.inserted_to_archive)
        assert inserted >= 1

    def test_archive_has_no_invalid_entries(self, kg):
        proposals = [_make_invalid_proposal(f"pinv{i}") for i in range(3)]
        result = run_pipeline(kg, proposals, seed=42)
        assert len(result.archive.cells) == 0

    def test_pipeline_result_has_archive_field(self, kg):
        proposals = [_make_valid_add_entity_proposal("p1")]
        result = run_pipeline(kg, proposals, seed=42)
        assert result.archive is not None
        assert hasattr(result.archive, "cells")
