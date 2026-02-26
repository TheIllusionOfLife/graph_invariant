"""Schema validation for Proposal objects."""

from __future__ import annotations

from harmony.proposals.types import Proposal, ProposalType, ValidationResult
from harmony.types import EdgeType

_VALID_EDGE_TYPE_NAMES: frozenset[str] = frozenset(et.name for et in EdgeType)
_MIN_TEXT_LEN = 10
_TEXT_FIELDS = ("claim", "justification", "falsification_condition")
_MIN_DOMAIN_LEN = 3


def validate(proposal: Proposal) -> ValidationResult:
    """Check schema compliance. All violations are collected before returning.

    Rules:
      1. Text fields (claim, justification, falsification_condition) ≥ 10 chars;
         kg_domain ≥ 3 chars (controlled vocabulary, not free text)
      2. ADD_EDGE / REMOVE_EDGE → source_entity, target_entity, edge_type required
         ADD_ENTITY             → entity_id, entity_type required
         REMOVE_ENTITY          → entity_id required
      3. edge_type ∈ {et.name for et in EdgeType} when present
    """
    violations: list[str] = []

    # Rule 1: text-field length
    for field_name in _TEXT_FIELDS:
        value: str | None = getattr(proposal, field_name)
        if not value or len(value) < _MIN_TEXT_LEN:
            actual = len(value) if value else 0
            violations.append(
                f"'{field_name}' must be at least {_MIN_TEXT_LEN} characters (got {actual})"
            )

    # Rule 1b: kg_domain — controlled vocabulary, lower minimum
    domain = proposal.kg_domain
    if not domain or len(domain) < _MIN_DOMAIN_LEN:
        actual = len(domain) if domain else 0
        violations.append(
            f"'kg_domain' must be at least {_MIN_DOMAIN_LEN} characters (got {actual})"
        )

    # Rule 2: type-specific required fields
    ptype = proposal.proposal_type
    if ptype in (ProposalType.ADD_EDGE, ProposalType.REMOVE_EDGE):
        if not proposal.source_entity:
            violations.append("'source_entity' is required for ADD_EDGE/REMOVE_EDGE")
        if not proposal.target_entity:
            violations.append("'target_entity' is required for ADD_EDGE/REMOVE_EDGE")
        if not proposal.edge_type:
            violations.append("'edge_type' is required for ADD_EDGE/REMOVE_EDGE")
    elif ptype == ProposalType.ADD_ENTITY:
        if not proposal.entity_id:
            violations.append("'entity_id' is required for ADD_ENTITY")
        if not proposal.entity_type:
            violations.append("'entity_type' is required for ADD_ENTITY")
    elif ptype == ProposalType.REMOVE_ENTITY:
        if not proposal.entity_id:
            violations.append("'entity_id' is required for REMOVE_ENTITY")

    # Rule 3: edge_type must be a known EdgeType name when actually supplied (non-empty)
    if proposal.edge_type and proposal.edge_type not in _VALID_EDGE_TYPE_NAMES:
        violations.append(
            f"'edge_type' must be one of {sorted(_VALID_EDGE_TYPE_NAMES)}, "
            f"got '{proposal.edge_type}'"
        )

    return ValidationResult(is_valid=len(violations) == 0, violations=violations)
