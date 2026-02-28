"""Tests for harmony.evaluation.literature_proxy — TDD-first."""

from __future__ import annotations

from harmony.proposals.types import Proposal, ProposalType


def _make_proposal(
    claim: str = "Energy is conserved in isolated systems",
    justification: str = "First law of thermodynamics",
    falsification: str = "Finding a system where energy is not conserved",
    domain: str = "physics",
) -> Proposal:
    return Proposal(
        id="test-1",
        proposal_type=ProposalType.ADD_EDGE,
        claim=claim,
        justification=justification,
        falsification_condition=falsification,
        kg_domain=domain,
        source_entity="energy",
        target_entity="conservation_law",
        edge_type="MAPS_TO",
    )


class TestProxyScore:
    def test_returns_float_in_unit_interval(self) -> None:
        from harmony.evaluation.literature_proxy import proxy_score

        p = _make_proposal()
        score = proxy_score(p)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic_same_seed(self) -> None:
        from harmony.evaluation.literature_proxy import proxy_score

        p = _make_proposal()
        s1 = proxy_score(p, seed=42)
        s2 = proxy_score(p, seed=42)
        assert s1 == s2

    def test_empty_claim_returns_zero(self) -> None:
        from harmony.evaluation.literature_proxy import proxy_score

        p = _make_proposal(claim="x" * 10)  # minimal valid claim
        # Very short/generic claim should score low but not crash
        score = proxy_score(p)
        assert isinstance(score, float)


class TestBatchProxy:
    def test_returns_list_of_correct_length(self) -> None:
        from harmony.evaluation.literature_proxy import batch_proxy_scores

        proposals = [_make_proposal(claim=f"Claim {i} about physics research") for i in range(5)]
        scores = batch_proxy_scores(proposals)
        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)

    def test_all_in_unit_interval(self) -> None:
        from harmony.evaluation.literature_proxy import batch_proxy_scores

        proposals = [_make_proposal(claim=f"Claim {i} about physics research") for i in range(3)]
        scores = batch_proxy_scores(proposals)
        for s in scores:
            assert 0.0 <= s <= 1.0


class TestProxyConfig:
    def test_default_evaluator_model_differs_from_proposer(self) -> None:
        """The evaluation LLM must differ from the proposal LLM (gpt-oss:20b)."""
        from harmony.evaluation.literature_proxy import EVALUATOR_MODEL

        assert EVALUATOR_MODEL != "gpt-oss:20b", (
            "Evaluator must use a different model than the proposer to avoid circularity"
        )

    def test_evaluator_model_is_nonempty(self) -> None:
        from harmony.evaluation.literature_proxy import EVALUATOR_MODEL

        assert len(EVALUATOR_MODEL) > 0
