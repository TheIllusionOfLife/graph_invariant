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
    """A node in the knowledge graph with a typed identity and property bag."""

    id: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (properties are shallow-copied)."""
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entity:
        """Deserialise from a dict produced by ``to_dict``."""
        return cls(id=d["id"], entity_type=d["entity_type"], properties=d.get("properties", {}))


@dataclass(slots=True)
class TypedEdge:
    """A directed, typed edge connecting two entities in the knowledge graph."""

    source: str
    target: str
    edge_type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (properties are shallow-copied)."""
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "properties": dict(self.properties),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TypedEdge:
        """Deserialise from a dict produced by ``to_dict``."""
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=EdgeType(d["edge_type"]),
            properties=d.get("properties", {}),
        )


@dataclass(slots=True)
class KnowledgeGraph:
    """A typed, directed knowledge graph with named entities and typed edges."""

    domain: str
    entities: dict[str, Entity] = field(default_factory=dict)
    edges: list[TypedEdge] = field(default_factory=list)

    @property
    def num_entities(self) -> int:
        """Number of entities in the graph."""
        return len(self.entities)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    def add_entity(self, entity: Entity) -> None:
        """Add an entity; raises ValueError on duplicate id."""
        if entity.id in self.entities:
            raise ValueError(f"Entity '{entity.id}' already exists in KG '{self.domain}'")
        self.entities[entity.id] = entity

    def add_edge(self, edge: TypedEdge) -> None:
        """Add an edge; raises ValueError if source or target entity is unknown."""
        if edge.source not in self.entities:
            raise ValueError(f"source entity '{edge.source}' not found in KG '{self.domain}'")
        if edge.target not in self.entities:
            raise ValueError(f"target entity '{edge.target}' not found in KG '{self.domain}'")
        self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "domain": self.domain,
            "entities": [e.to_dict() for e in self.entities.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeGraph:
        """Deserialise from a dict; validates referential integrity via add_entity/add_edge."""
        kg = cls(domain=d["domain"])
        for e in d["entities"]:
            kg.add_entity(Entity.from_dict(e))
        for e in d["edges"]:
            kg.add_edge(TypedEdge.from_dict(e))
        return kg
