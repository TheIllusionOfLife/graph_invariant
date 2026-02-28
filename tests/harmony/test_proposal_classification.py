"""Tests for analysis.proposal_classification — TDD-first."""

from __future__ import annotations

from harmony.proposals.types import Proposal, ProposalType
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(domain="test")
    kg.add_entity(Entity(id="A", entity_type="concept"))
    kg.add_entity(Entity(id="B", entity_type="concept"))
    kg.add_entity(Entity(id="C", entity_type="concept"))
    kg.add_edge(TypedEdge(source="A", target="B", edge_type=EdgeType.DEPENDS_ON))
    return kg


def _make_proposal(source: str, target: str, edge_type: str = "DEPENDS_ON") -> Proposal:
    return Proposal(
        id=f"{source}-{target}",
        proposal_type=ProposalType.ADD_EDGE,
        claim=f"{source} depends on {target}",
        justification="Theoretical reasoning about the relationship",
        falsification_condition="Finding no dependency between them",
        kg_domain="test",
        source_entity=source,
        target_entity=target,
        edge_type=edge_type,
    )


class TestClassifyProposal:
    def test_existing_edge_is_rediscovery(self) -> None:
        from analysis.proposal_classification import classify_proposal

        kg = _make_kg()
        p = _make_proposal("A", "B", "DEPENDS_ON")
        assert classify_proposal(p, kg) == "rediscovery"

    def test_novel_edge_is_novel(self) -> None:
        from analysis.proposal_classification import classify_proposal

        kg = _make_kg()
        p = _make_proposal("A", "C", "DEPENDS_ON")
        assert classify_proposal(p, kg) == "novel"

    def test_reversed_edge_is_novel(self) -> None:
        from analysis.proposal_classification import classify_proposal

        kg = _make_kg()
        p = _make_proposal("B", "A", "DEPENDS_ON")
        assert classify_proposal(p, kg) == "novel"

    def test_different_edge_type_is_novel(self) -> None:
        from analysis.proposal_classification import classify_proposal

        kg = _make_kg()
        p = _make_proposal("A", "B", "GENERALIZES")
        assert classify_proposal(p, kg) == "novel"


class TestClassifyBatch:
    def test_returns_dict_with_correct_counts(self) -> None:
        from analysis.proposal_classification import classify_batch

        kg = _make_kg()
        proposals = [
            _make_proposal("A", "B", "DEPENDS_ON"),  # rediscovery
            _make_proposal("A", "C", "DEPENDS_ON"),  # novel
            _make_proposal("B", "C", "MAPS_TO"),  # novel
        ]
        result = classify_batch(proposals, kg)
        assert result["rediscovery"] == 1
        assert result["novel"] == 2
        assert result["total"] == 3
