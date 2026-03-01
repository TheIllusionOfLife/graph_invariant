"""Tests for analysis.falsification_quality — testability of falsification conditions."""

from __future__ import annotations

from harmony.proposals.types import Proposal, ProposalType


def _make_proposal(
    falsification: str,
    claim: str = "Test claim about the relationship",
) -> Proposal:
    return Proposal(
        id="test-1",
        proposal_type=ProposalType.ADD_EDGE,
        claim=claim,
        justification="Theoretical reasoning about the relationship",
        falsification_condition=falsification,
        kg_domain="test",
        source_entity="A",
        target_entity="B",
        edge_type="DEPENDS_ON",
    )


class TestFalsificationScore:
    def test_specific_condition_scores_higher(self) -> None:
        from analysis.falsification_quality import falsification_score

        specific = _make_proposal(
            "If measuring thermal conductivity of copper at 300K yields values "
            "below 350 W/mK, the dependency claim is falsified."
        )
        vague = _make_proposal("If this is wrong somehow.")
        assert falsification_score(specific) > falsification_score(vague)

    def test_score_in_unit_interval(self) -> None:
        from analysis.falsification_quality import falsification_score

        p = _make_proposal("If X does not cause Y in controlled experiments.")
        score = falsification_score(p)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_empty_condition_scores_low(self) -> None:
        from analysis.falsification_quality import falsification_score

        p = _make_proposal("")
        assert falsification_score(p) < 0.2


class TestBatchFalsification:
    def test_returns_correct_length(self) -> None:
        from analysis.falsification_quality import batch_falsification_scores

        proposals = [
            _make_proposal("If X fails under condition Y."),
            _make_proposal("If experiment Z produces result W."),
        ]
        scores = batch_falsification_scores(proposals)
        assert len(scores) == 2

    def test_all_scores_in_unit_interval(self) -> None:
        from analysis.falsification_quality import batch_falsification_scores

        proposals = [
            _make_proposal("If X fails under condition Y."),
            _make_proposal("If experiment Z produces result W."),
        ]
        scores = batch_falsification_scores(proposals)
        for s in scores:
            assert 0.0 <= s <= 1.0


class TestTestabilityFraction:
    def test_fraction_is_ratio(self) -> None:
        from analysis.falsification_quality import testability_fraction

        proposals = [
            _make_proposal(  # testable — specific
                "If spectroscopic analysis of the compound shows no absorption "
                "peak at 450nm, the claim is falsified."
            ),
            _make_proposal("Wrong somehow."),  # not testable — too vague
        ]
        frac = testability_fraction(proposals, threshold=0.4)
        assert isinstance(frac, float)
        assert 0.0 <= frac <= 1.0

    def test_empty_list_returns_zero(self) -> None:
        from analysis.falsification_quality import testability_fraction

        assert testability_fraction([], threshold=0.4) == 0.0
