"""Tests for harmony.proposals.types — schema fields and round-trip serialisation.

TDD: these tests are written BEFORE implementation. They verify:
  - ProposalType has exactly 4 variants
  - Proposal dataclass has all required and optional fields
  - ValidationResult has is_valid and violations fields
  - Proposal round-trips through to_dict() / from_dict()
"""

from __future__ import annotations

from dataclasses import fields

from harmony.proposals.types import Proposal, ProposalType, ValidationResult


def _make_valid_add_edge_proposal() -> Proposal:
    return Proposal(
        id="test-001",
        proposal_type=ProposalType.ADD_EDGE,
        claim="Eigenvectors depend on the determinant in a fundamental way.",
        justification="This follows from the characteristic polynomial definition.",
        falsification_condition="If a matrix has zero determinant but non-trivial eigenvectors.",
        kg_domain="linear_algebra",
        source_entity="eigenvector",
        target_entity="determinant",
        edge_type="DEPENDS_ON",
    )


class TestProposalTypeEnum:
    def test_proposal_type_has_4_variants(self):
        variants = {pt.value for pt in ProposalType}
        assert variants == {"add_edge", "remove_edge", "add_entity", "remove_entity"}

    def test_proposal_type_values_are_strings(self):
        for pt in ProposalType:
            assert isinstance(pt.value, str)


class TestProposalDataclass:
    def test_proposal_dataclass_has_required_fields(self):
        field_names = {f.name for f in fields(Proposal)}
        required = {
            "id",
            "proposal_type",
            "claim",
            "justification",
            "falsification_condition",
            "kg_domain",
        }
        assert required.issubset(field_names)

    def test_proposal_dataclass_has_optional_mutation_fields(self):
        field_names = {f.name for f in fields(Proposal)}
        optional = {"source_entity", "target_entity", "edge_type", "entity_id", "entity_type"}
        assert optional.issubset(field_names)

    def test_optional_fields_default_to_none(self):
        p = Proposal(
            id="test",
            proposal_type=ProposalType.ADD_ENTITY,
            claim="A concept exists here in this domain.",
            justification="We observe it consistently in experiments.",
            falsification_condition="If the concept cannot be identified experimentally.",
            kg_domain="physics_domain",
        )
        assert p.source_entity is None
        assert p.target_entity is None
        assert p.edge_type is None
        assert p.entity_id is None
        assert p.entity_type is None

    def test_proposal_stores_all_add_edge_fields(self):
        p = _make_valid_add_edge_proposal()
        assert p.id == "test-001"
        assert p.proposal_type == ProposalType.ADD_EDGE
        assert p.source_entity == "eigenvector"
        assert p.target_entity == "determinant"
        assert p.edge_type == "DEPENDS_ON"


class TestValidationResultDataclass:
    def test_validation_result_fields(self):
        vr = ValidationResult(is_valid=True, violations=[])
        assert vr.is_valid is True
        assert vr.violations == []

    def test_validation_result_invalid_with_violations(self):
        violations = ["claim too short", "missing edge_type"]
        vr = ValidationResult(is_valid=False, violations=violations)
        assert vr.is_valid is False
        assert len(vr.violations) == 2

    def test_validation_result_violations_are_strings(self):
        vr = ValidationResult(is_valid=False, violations=["first violation"])
        assert all(isinstance(v, str) for v in vr.violations)


class TestProposalRoundTrip:
    def test_proposal_round_trips_dict(self):
        p = _make_valid_add_edge_proposal()
        d = p.to_dict()
        p2 = Proposal.from_dict(d)
        assert p2.id == p.id
        assert p2.proposal_type == p.proposal_type
        assert p2.claim == p.claim
        assert p2.justification == p.justification
        assert p2.falsification_condition == p.falsification_condition
        assert p2.kg_domain == p.kg_domain
        assert p2.source_entity == p.source_entity
        assert p2.target_entity == p.target_entity
        assert p2.edge_type == p.edge_type
        assert p2.entity_id == p.entity_id
        assert p2.entity_type == p.entity_type

    def test_to_dict_serialises_proposal_type_as_string(self):
        p = _make_valid_add_edge_proposal()
        d = p.to_dict()
        assert isinstance(d["proposal_type"], str)
        assert d["proposal_type"] == "add_edge"

    def test_to_dict_contains_all_keys(self):
        p = _make_valid_add_edge_proposal()
        d = p.to_dict()
        expected_keys = {
            "id",
            "proposal_type",
            "claim",
            "justification",
            "falsification_condition",
            "kg_domain",
            "source_entity",
            "target_entity",
            "edge_type",
            "entity_id",
            "entity_type",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_handles_optional_none_fields(self):
        d = {
            "id": "test-002",
            "proposal_type": "add_entity",
            "claim": "A new entity represents this concept clearly.",
            "justification": "The concept is not yet in the graph here.",
            "falsification_condition": "If the entity is redundant with an existing one.",
            "kg_domain": "astronomy_domain",
            "entity_id": "new_concept",
            "entity_type": "concept",
            "source_entity": None,
            "target_entity": None,
            "edge_type": None,
        }
        p = Proposal.from_dict(d)
        assert p.source_entity is None
        assert p.target_entity is None
        assert p.edge_type is None
        assert p.entity_id == "new_concept"
        assert p.entity_type == "concept"

    def test_round_trip_preserves_none_optional_fields(self):
        p = Proposal(
            id="test-003",
            proposal_type=ProposalType.REMOVE_ENTITY,
            claim="This entity is redundant and should be merged elsewhere.",
            justification="It duplicates another concept already in knowledge graph.",
            falsification_condition="If removing it breaks any downstream theoretical chain.",
            kg_domain="physics_domain",
            entity_id="old_concept",
        )
        p2 = Proposal.from_dict(p.to_dict())
        assert p2.source_entity is None
        assert p2.target_entity is None
        assert p2.edge_type is None
        assert p2.entity_type is None
