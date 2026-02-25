"""Proposal schema types for structured KG mutations with theory metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProposalType(Enum):
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ADD_ENTITY = "add_entity"
    REMOVE_ENTITY = "remove_entity"


@dataclass
class Proposal:
    """A structured KG mutation paired with its theoretical justification."""

    id: str
    proposal_type: ProposalType
    claim: str  # 1-sentence theoretical claim (≥10 chars)
    justification: str  # reasoning (≥10 chars)
    falsification_condition: str  # what would disprove it (≥10 chars)
    kg_domain: str  # target domain (≥10 chars)

    # Mutation parameters — required depending on proposal_type
    source_entity: str | None = None  # ADD_EDGE / REMOVE_EDGE
    target_entity: str | None = None  # ADD_EDGE / REMOVE_EDGE
    edge_type: str | None = None  # ADD_EDGE / REMOVE_EDGE — must be valid EdgeType.name
    entity_id: str | None = None  # ADD_ENTITY / REMOVE_ENTITY
    entity_type: str | None = None  # ADD_ENTITY

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "id": self.id,
            "proposal_type": self.proposal_type.value,
            "claim": self.claim,
            "justification": self.justification,
            "falsification_condition": self.falsification_condition,
            "kg_domain": self.kg_domain,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "edge_type": self.edge_type,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Proposal:
        """Deserialise from a dict produced by ``to_dict``."""
        return cls(
            id=d["id"],
            proposal_type=ProposalType(d["proposal_type"]),
            claim=d["claim"],
            justification=d["justification"],
            falsification_condition=d["falsification_condition"],
            kg_domain=d["kg_domain"],
            source_entity=d.get("source_entity"),
            target_entity=d.get("target_entity"),
            edge_type=d.get("edge_type"),
            entity_id=d.get("entity_id"),
            entity_type=d.get("entity_type"),
        )


@dataclass
class ValidationResult:
    """Outcome of schema validation for a Proposal."""

    is_valid: bool
    violations: list[str]  # human-readable; empty when is_valid=True
