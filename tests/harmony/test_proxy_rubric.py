"""Tests for harmony.evaluation.proxy_rubric — TDD-first."""

from __future__ import annotations

from harmony.proposals.types import Proposal, ProposalType


def _make_proposal(
    claim: str = "Superconductivity requires Cooper pair formation",
    justification: str = "BCS theory predicts electron pairing via phonon exchange",
    falsification: str = "Finding superconductivity without any pair condensate",
    domain: str = "materials",
) -> Proposal:
    return Proposal(
        id="test-rubric-1",
        proposal_type=ProposalType.ADD_EDGE,
        claim=claim,
        justification=justification,
        falsification_condition=falsification,
        kg_domain=domain,
        source_entity="superconductivity",
        target_entity="bcs_theory",
        edge_type="EXPLAINS",
    )


class TestRubricScore:
    def test_returns_dict_with_five_criteria(self) -> None:
        from harmony.evaluation.proxy_rubric import rubric_score

        p = _make_proposal()
        result = rubric_score(p)
        assert isinstance(result, dict)
        expected_keys = {
            "novelty",
            "plausibility",
            "specificity",
            "testability",
            "relevance",
        }
        assert set(result.keys()) == expected_keys

    def test_all_scores_in_1_to_5(self) -> None:
        from harmony.evaluation.proxy_rubric import rubric_score

        p = _make_proposal()
        result = rubric_score(p)
        for criterion, score in result.items():
            assert isinstance(score, (int, float)), f"{criterion} not numeric"
            assert 1 <= score <= 5, f"{criterion} = {score} out of [1,5]"

    def test_deterministic_same_seed(self) -> None:
        from harmony.evaluation.proxy_rubric import rubric_score

        p = _make_proposal()
        r1 = rubric_score(p, seed=42)
        r2 = rubric_score(p, seed=42)
        assert r1 == r2


class TestRubricAggregate:
    def test_aggregate_is_mean_of_criteria(self) -> None:
        from harmony.evaluation.proxy_rubric import rubric_aggregate, rubric_score

        p = _make_proposal()
        scores = rubric_score(p, seed=42)
        agg = rubric_aggregate(scores)
        expected = sum(scores.values()) / len(scores)
        assert abs(agg - expected) < 1e-9

    def test_aggregate_in_1_to_5(self) -> None:
        from harmony.evaluation.proxy_rubric import rubric_aggregate, rubric_score

        p = _make_proposal()
        scores = rubric_score(p, seed=42)
        agg = rubric_aggregate(scores)
        assert 1.0 <= agg <= 5.0


class TestBatchRubric:
    def test_batch_returns_list_of_dicts(self) -> None:
        from harmony.evaluation.proxy_rubric import batch_rubric_scores

        proposals = [_make_proposal(claim=f"Claim {i} about materials") for i in range(3)]
        results = batch_rubric_scores(proposals)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, dict)
            assert len(r) == 5


class TestRubricCriteria:
    def test_criteria_definitions_exist(self) -> None:
        from harmony.evaluation.proxy_rubric import RUBRIC_CRITERIA

        assert len(RUBRIC_CRITERIA) == 5
        for name, description in RUBRIC_CRITERIA.items():
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(description) > 10
