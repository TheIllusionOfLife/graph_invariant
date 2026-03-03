"""Tests for factor decomposition baselines.

TDD: written BEFORE implementation. Verifies:
  - HarmonyConfig accepts accept_all_valid and greedy flags
  - random_proposer generates valid proposals deterministically
  - Factor decomposition configs create valid HarmonyConfig instances
"""

from __future__ import annotations

from harmony.config import HarmonyConfig
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge


def _make_test_kg() -> KnowledgeGraph:
    """Small KG for testing random proposer."""
    kg = KnowledgeGraph(domain="test")
    for i in range(10):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    for i in range(10):
        kg.add_edge(
            TypedEdge(
                source=f"e{i}",
                target=f"e{(i + 1) % 10}",
                edge_type=EdgeType.DEPENDS_ON,
            )
        )
    return kg


class TestConfigFlags:
    def test_accept_all_valid_default_false(self):
        cfg = HarmonyConfig()
        assert cfg.accept_all_valid is False

    def test_greedy_default_false(self):
        cfg = HarmonyConfig()
        assert cfg.greedy is False

    def test_accept_all_valid_true(self):
        cfg = HarmonyConfig(accept_all_valid=True)
        assert cfg.accept_all_valid is True

    def test_greedy_true(self):
        cfg = HarmonyConfig(greedy=True)
        assert cfg.greedy is True

    def test_both_flags_true(self):
        """Both flags can be set simultaneously (though unusual)."""
        cfg = HarmonyConfig(accept_all_valid=True, greedy=True)
        assert cfg.accept_all_valid is True
        assert cfg.greedy is True


class TestRandomProposer:
    def test_generates_correct_count(self):
        from harmony.proposals.random_proposer import generate_random_proposals

        kg = _make_test_kg()
        proposals = generate_random_proposals(kg, n=5, seed=42)
        assert len(proposals) == 5

    def test_proposals_are_valid_schema(self):
        from harmony.proposals.random_proposer import generate_random_proposals
        from harmony.proposals.types import ProposalType

        kg = _make_test_kg()
        proposals = generate_random_proposals(kg, n=3, seed=42)
        for p in proposals:
            assert p.proposal_type == ProposalType.ADD_EDGE
            assert p.source_entity in kg.entities
            assert p.target_entity in kg.entities
            assert p.edge_type in [et.name for et in EdgeType]

    def test_deterministic_with_same_seed(self):
        from harmony.proposals.random_proposer import generate_random_proposals

        kg = _make_test_kg()
        a = generate_random_proposals(kg, n=5, seed=42)
        b = generate_random_proposals(kg, n=5, seed=42)
        for pa, pb in zip(a, b, strict=True):
            assert pa.source_entity == pb.source_entity
            assert pa.target_entity == pb.target_entity
            assert pa.edge_type == pb.edge_type

    def test_different_seeds_differ(self):
        from harmony.proposals.random_proposer import generate_random_proposals

        kg = _make_test_kg()
        a = generate_random_proposals(kg, n=5, seed=42)
        b = generate_random_proposals(kg, n=5, seed=99)
        # At least one proposal should differ
        any_different = any(
            pa.source_entity != pb.source_entity or pa.target_entity != pb.target_entity
            for pa, pb in zip(a, b, strict=True)
        )
        assert any_different


class TestFactorDecompConfigs:
    def test_llm_only_config(self):
        """LLM-only: accept_all_valid=True, no Harmony selection."""
        cfg = HarmonyConfig(accept_all_valid=True)
        assert cfg.accept_all_valid is True
        assert cfg.greedy is False

    def test_no_qd_config(self):
        """No-QD: greedy=True, single-bin archive."""
        cfg = HarmonyConfig(greedy=True)
        assert cfg.greedy is True
        assert cfg.accept_all_valid is False
