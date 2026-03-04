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
            pa.source_entity != pb.source_entity
            or pa.target_entity != pb.target_entity
            or pa.edge_type != pb.edge_type
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


class TestPipelineAcceptAllValid:
    """TDD: verify accept_all_valid skips harmony_score in pipeline."""

    def test_pipeline_accept_all_valid_skips_scoring(self, monkeypatch):
        """When accept_all_valid=True, harmony_score is never called."""
        from harmony.proposals import pipeline as pipeline_mod
        from harmony.proposals.pipeline import run_pipeline
        from harmony.proposals.types import Proposal, ProposalType

        # If harmony_score is called, this will blow up
        def _boom(*args, **kwargs):
            raise AssertionError("harmony_score should not be called")

        monkeypatch.setattr(pipeline_mod, "harmony_score", _boom)

        kg = _make_test_kg()
        proposal = Proposal(
            id="test-001",
            proposal_type=ProposalType.ADD_EDGE,
            claim="Test claim for accept_all_valid pipeline.",
            justification="Testing that scoring is skipped entirely.",
            falsification_condition="If harmony_score is called, the test fails.",
            kg_domain="test",
            source_entity="e0",
            target_entity="e5",
            edge_type="DEPENDS_ON",
        )

        result = run_pipeline(
            kg=kg,
            proposals=[proposal],
            seed=42,
            archive_bins=5,
            accept_all_valid=True,
        )
        assert result.valid_rate == 1.0
        assert len(result.results) == 1
        assert result.results[0].harmony_gain == 1.0
        assert result.results[0].inserted_to_archive is True

    def test_pipeline_accept_all_valid_false_default(self):
        """Default (accept_all_valid=False) still computes harmony_score."""
        from harmony.proposals.pipeline import run_pipeline
        from harmony.proposals.types import Proposal, ProposalType

        kg = _make_test_kg()
        proposal = Proposal(
            id="test-002",
            proposal_type=ProposalType.ADD_EDGE,
            claim="Test claim for default pipeline behavior.",
            justification="Default pipeline should compute harmony_score.",
            falsification_condition="If harmony_score is not computed, test fails.",
            kg_domain="test",
            source_entity="e0",
            target_entity="e5",
            edge_type="DEPENDS_ON",
        )

        result = run_pipeline(kg=kg, proposals=[proposal], seed=42, archive_bins=5)
        assert result.valid_rate == 1.0
        # Default should produce a real gain (not 1.0 constant)
        assert result.results[0].harmony_gain is not None


class TestHarmonyLoopForwardsFlags:
    """TDD: verify harmony_loop forwards accept_all_valid to run_pipeline."""

    def test_harmony_loop_forwards_accept_all_valid(self, monkeypatch):
        """run_harmony_loop passes cfg.accept_all_valid to run_pipeline."""
        import harmony.harmony_loop as loop_mod

        captured_kwargs: list[dict] = []

        def _capture_pipeline(*args, **kwargs):
            captured_kwargs.append(kwargs)
            # Return a minimal result to keep the loop going
            from harmony.map_elites import HarmonyMapElites
            from harmony.proposals.pipeline import PipelineResult

            return PipelineResult(
                results=[],
                valid_rate=0.0,
                archive=HarmonyMapElites(num_bins=kwargs.get("archive_bins", 5)),
            )

        # Patch both run_pipeline and _run_island_generation to avoid LLM calls
        monkeypatch.setattr(loop_mod, "run_pipeline", _capture_pipeline)
        monkeypatch.setattr(
            loop_mod,
            "_run_island_generation",
            lambda **kw: ([], 0),
        )

        cfg = HarmonyConfig(
            accept_all_valid=True,
            max_generations=1,
            island_temperatures=(0.5,),
        )
        kg = _make_test_kg()

        from harmony.harmony_loop import run_harmony_loop

        run_harmony_loop(cfg, kg, output_dir="/tmp/test_forward_flag")

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0]["accept_all_valid"] is True


class TestBackendConfig:
    def test_backend_default_is_ollama(self):
        cfg = HarmonyConfig()
        assert cfg.backend == "ollama"

    def test_backend_mlx_accepted(self):
        cfg = HarmonyConfig(backend="mlx")
        assert cfg.backend == "mlx"

    def test_backend_invalid_raises(self):
        import pytest

        with pytest.raises(ValueError, match="backend must be"):
            HarmonyConfig(backend="invalid")

    def test_mlx_model_id_default(self):
        cfg = HarmonyConfig()
        assert cfg.mlx_model_id == "mlx-community/Qwen3.5-35B-A3B-4bit"


class TestCLIFlags:
    """TDD: verify CLI parses accept-all-valid and greedy flags."""

    def test_parse_accept_all_valid_flag(self):
        from harmony.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--accept-all-valid"])
        assert args.accept_all_valid is True

    def test_parse_greedy_flag(self):
        from harmony.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["--greedy"])
        assert args.greedy is True

    def test_default_flags_false(self):
        from harmony.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.accept_all_valid is False
        assert args.greedy is False
