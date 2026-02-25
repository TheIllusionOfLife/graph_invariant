"""KGDataset: edge splitting, masking, and hidden-edge backtesting scaffold."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .types import Entity, KnowledgeGraph, TypedEdge


@dataclass
class KGDataset:
    """Wraps a KnowledgeGraph with train/val/test/hidden edge splits.

    10% of edges are reserved as the ground-truth recoverable set for backtesting.
    The remaining 90% are split 80/10/10 into train/val/test.
    """

    kg: KnowledgeGraph
    train_edges: list[TypedEdge] = field(default_factory=list)
    val_edges: list[TypedEdge] = field(default_factory=list)
    test_edges: list[TypedEdge] = field(default_factory=list)
    hidden_edges: list[TypedEdge] = field(default_factory=list)

    @classmethod
    def from_kg(
        cls,
        kg: KnowledgeGraph,
        seed: int = 42,
        hidden_ratio: float = 0.1,
    ) -> KGDataset:
        """Split KG edges into train/val/test and hidden backtesting set.

        Args:
            kg: The source knowledge graph.
            seed: RNG seed for deterministic splits.
            hidden_ratio: Fraction of edges to reserve as ground-truth (default 0.1).
        """
        if not (0.0 < hidden_ratio < 1.0):
            raise ValueError(f"hidden_ratio must be in (0, 1), got {hidden_ratio}")

        edges = list(kg.edges)
        rng = random.Random(seed)
        rng.shuffle(edges)

        n_total = len(edges)
        n_hidden = max(1, round(n_total * hidden_ratio))
        n_remaining = n_total - n_hidden

        # Of remaining edges: 80% train, 10% val, 10% test
        n_val = max(0, round(n_remaining * 0.1))
        n_test = max(0, round(n_remaining * 0.1))
        n_train = max(0, n_remaining - n_val - n_test)

        hidden = edges[:n_hidden]
        remaining = edges[n_hidden:]

        train = remaining[:n_train]
        val = remaining[n_train : n_train + n_val]
        test = remaining[n_train + n_val :]

        return cls(
            kg=kg,
            train_edges=train,
            val_edges=val,
            test_edges=test,
            hidden_edges=hidden,
        )

    def train_kg(self) -> KnowledgeGraph:
        """Return a KG containing all entities but only train edges (no hidden/val/test).

        Iterates self.train_edges directly to avoid mis-excluding parallel edges that
        share the same (source, target, edge_type) triple but differ in properties.
        """
        masked = KnowledgeGraph(domain=self.kg.domain)
        for entity in self.kg.entities.values():
            masked.add_entity(
                Entity(
                    id=entity.id,
                    entity_type=entity.entity_type,
                    properties=dict(entity.properties),
                )
            )
        for edge in self.train_edges:
            masked.add_edge(
                TypedEdge(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    properties=dict(edge.properties),
                )
            )
        return masked

    def to_dict(self) -> dict[str, Any]:
        def _edges(lst: list[TypedEdge]) -> list[dict[str, Any]]:
            return [e.to_dict() for e in lst]

        return {
            "kg": self.kg.to_dict(),
            "train_edges": _edges(self.train_edges),
            "val_edges": _edges(self.val_edges),
            "test_edges": _edges(self.test_edges),
            "hidden_edges": _edges(self.hidden_edges),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KGDataset:
        """Deserialise from a dict produced by ``to_dict``."""
        kg = KnowledgeGraph.from_dict(d["kg"])
        return cls(
            kg=kg,
            train_edges=[TypedEdge.from_dict(e) for e in d["train_edges"]],
            val_edges=[TypedEdge.from_dict(e) for e in d["val_edges"]],
            test_edges=[TypedEdge.from_dict(e) for e in d["test_edges"]],
            hidden_edges=[TypedEdge.from_dict(e) for e in d["hidden_edges"]],
        )
