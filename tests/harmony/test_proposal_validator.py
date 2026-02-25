"""Tests for harmony.proposals.validator — one test per validation rule.

TDD: these tests are written BEFORE implementation. They verify:
  - Valid proposals pass without violations
  - Each short-text field generates a specific violation
  - Type-specific required fields are checked per ProposalType
  - Invalid edge_type values are caught
  - Violation list is empty on success and cumulative on failure
"""

from __future__ import annotations

import pytest

from harmony.proposals.types import Proposal, ProposalType
from harmony.proposals.validator import validate


def _make_add_edge_proposal(**overrides) -> Proposal:
    defaults = dict(
        id="test-add-edge",
        proposal_type=ProposalType.ADD_EDGE,
        claim="Eigenvectors depend on the determinant in a fundamental way.",
        justification="This follows from the characteristic polynomial definition.",
        falsification_condition="If a matrix has zero determinant but non-trivial eigenvectors.",
        kg_domain="linear_algebra",
        source_entity="eigenvector",
        target_entity="determinant",
        edge_type="DEPENDS_ON",
    )
    defaults.update(overrides)
    return Proposal(**defaults)


def _make_add_entity_proposal(**overrides) -> Proposal:
    defaults = dict(
        id="test-add-entity",
        proposal_type=ProposalType.ADD_ENTITY,
        claim="A new mathematical concept can be formalized in this domain.",
        justification="It is observed in multiple proofs and theorems consistently.",
        falsification_condition="If the concept reduces to an existing one without loss.",
        kg_domain="linear_algebra",
        entity_id="new_concept",
        entity_type="concept",
    )
    defaults.update(overrides)
    return Proposal(**defaults)


def _make_remove_entity_proposal(**overrides) -> Proposal:
    defaults = dict(
        id="test-remove-entity",
        proposal_type=ProposalType.REMOVE_ENTITY,
        claim="This entity is redundant and should be removed from the graph.",
        justification="It duplicates another concept already in the knowledge graph.",
        falsification_condition="If the entity provides unique information not elsewhere.",
        kg_domain="linear_algebra",
        entity_id="redundant_concept",
    )
    defaults.update(overrides)
    return Proposal(**defaults)


class TestValidAddEdge:
    def test_valid_add_edge_passes(self):
        p = _make_add_edge_proposal()
        result = validate(p)
        assert result.is_valid is True
        assert result.violations == []


class TestTextFieldValidation:
    def test_short_claim_fails(self):
        p = _make_add_edge_proposal(claim="hi")
        result = validate(p)
        assert result.is_valid is False
        assert any("claim" in v for v in result.violations)

    def test_missing_justification_fails(self):
        p = _make_add_edge_proposal(justification="short")
        result = validate(p)
        assert result.is_valid is False
        assert any("justification" in v for v in result.violations)

    def test_missing_falsification_fails(self):
        p = _make_add_edge_proposal(falsification_condition="no")
        result = validate(p)
        assert result.is_valid is False
        assert any("falsification_condition" in v for v in result.violations)

    def test_short_kg_domain_fails(self):
        p = _make_add_edge_proposal(kg_domain="la")
        result = validate(p)
        assert result.is_valid is False
        assert any("kg_domain" in v for v in result.violations)


class TestAddEdgeTypeSpecific:
    def test_add_edge_missing_source_fails(self):
        p = _make_add_edge_proposal(source_entity=None)
        result = validate(p)
        assert result.is_valid is False
        assert any("source_entity" in v for v in result.violations)

    def test_add_edge_missing_target_fails(self):
        p = _make_add_edge_proposal(target_entity=None)
        result = validate(p)
        assert result.is_valid is False
        assert any("target_entity" in v for v in result.violations)

    def test_add_edge_invalid_edge_type_fails(self):
        p = _make_add_edge_proposal(edge_type="INVALID_TYPE")
        result = validate(p)
        assert result.is_valid is False
        assert any("edge_type" in v for v in result.violations)

    def test_add_edge_missing_edge_type_fails(self):
        p = _make_add_edge_proposal(edge_type=None)
        result = validate(p)
        assert result.is_valid is False
        assert any("edge_type" in v for v in result.violations)

    def test_remove_edge_same_rules_as_add_edge(self):
        p = Proposal(
            id="test-remove-edge",
            proposal_type=ProposalType.REMOVE_EDGE,
            claim="Eigenvectors depend on the determinant in a fundamental way.",
            justification="This follows from the characteristic polynomial definition.",
            falsification_condition="If a matrix has zero determinant but non-trivial eigenvectors.",
            kg_domain="linear_algebra",
            source_entity="eigenvector",
            target_entity="determinant",
            edge_type="DEPENDS_ON",
        )
        result = validate(p)
        assert result.is_valid is True


class TestAddEntityTypeSpecific:
    def test_add_entity_missing_entity_type_fails(self):
        p = _make_add_entity_proposal(entity_type=None)
        result = validate(p)
        assert result.is_valid is False
        assert any("entity_type" in v for v in result.violations)

    def test_add_entity_missing_entity_id_fails(self):
        p = _make_add_entity_proposal(entity_id=None)
        result = validate(p)
        assert result.is_valid is False
        assert any("entity_id" in v for v in result.violations)

    def test_valid_add_entity_passes(self):
        p = _make_add_entity_proposal()
        result = validate(p)
        assert result.is_valid is True


class TestRemoveEntityTypeSpecific:
    def test_remove_entity_only_needs_entity_id(self):
        p = _make_remove_entity_proposal()
        result = validate(p)
        assert result.is_valid is True

    def test_remove_entity_missing_entity_id_fails(self):
        p = _make_remove_entity_proposal(entity_id=None)
        result = validate(p)
        assert result.is_valid is False
        assert any("entity_id" in v for v in result.violations)


class TestViolationList:
    def test_violation_list_empty_on_success(self):
        p = _make_add_edge_proposal()
        result = validate(p)
        assert result.violations == []

    def test_multiple_violations_collected(self):
        """All violations are collected, not just the first."""
        p = _make_add_edge_proposal(claim="hi", justification="no")
        result = validate(p)
        assert len(result.violations) >= 2

    def test_violations_are_human_readable_strings(self):
        p = _make_add_edge_proposal(claim="hi")
        result = validate(p)
        for v in result.violations:
            assert isinstance(v, str)
            assert len(v) > 0
