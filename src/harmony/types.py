"""Core typed knowledge graph types for the Harmony framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EdgeType(Enum):
    DEPENDS_ON = "depends_on"
    DERIVES = "derives"
    EQUIVALENT_TO = "equivalent_to"
    MAPS_TO = "maps_to"
    EXPLAINS = "explains"
    CONTRADICTS = "contradicts"
    GENERALIZES = "generalizes"


@dataclass(slots=True)
class Entity:
    id: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "entity_type": self.entity_type, "properties": self.properties}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entity:
        return cls(id=d["id"], entity_type=d["entity_type"], properties=d.get("properties", {}))


@dataclass(slots=True)
class TypedEdge:
    source: str
    target: str
    edge_type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TypedEdge:
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=EdgeType(d["edge_type"]),
            properties=d.get("properties", {}),
        )


@dataclass
class KnowledgeGraph:
    domain: str
    entities: dict[str, Entity] = field(default_factory=dict)
    edges: list[TypedEdge] = field(default_factory=list)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self.entities:
            raise ValueError(f"Entity '{entity.id}' already exists in KG '{self.domain}'")
        self.entities[entity.id] = entity

    def add_edge(self, edge: TypedEdge) -> None:
        if edge.source not in self.entities:
            raise ValueError(f"source entity '{edge.source}' not found in KG '{self.domain}'")
        if edge.target not in self.entities:
            raise ValueError(f"target entity '{edge.target}' not found in KG '{self.domain}'")
        self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "entities": [e.to_dict() for e in self.entities.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeGraph:
        kg = cls(domain=d["domain"])
        for e in d["entities"]:
            kg.entities[e["id"]] = Entity.from_dict(e)
        for e in d["edges"]:
            kg.edges.append(TypedEdge.from_dict(e))
        return kg
