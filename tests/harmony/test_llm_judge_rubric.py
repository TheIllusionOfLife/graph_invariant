"""Tests for scripts/llm_judge_rubric.py — TDD written before implementation.

Verifies:
  - build_rubric_prompt returns well-formed string for both variants
  - parse_rubric_scores extracts valid scores from mock LLM responses
  - compute_agreement calculates inter-prompt agreement correctly
  - aggregate_scores computes per-domain means
  - score validation enforces [1, 5] range
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


SAMPLE_PROPOSAL = {
    "id": "test-001",
    "claim": "Stellar nucleosynthesis explains the abundance of heavy elements.",
    "justification": "The r-process in supernovae produces heavy nuclei.",
    "falsification_condition": "Finding heavy elements without any stellar source.",
    "source_entity": "stellar_nucleosynthesis",
    "target_entity": "heavy_element_abundance",
    "edge_type": "explains",
    "kg_domain": "astronomy",
    "proposal_type": "ADD_EDGE",
}


# ---------------------------------------------------------------------------
# build_rubric_prompt
# ---------------------------------------------------------------------------


class TestBuildRubricPrompt:
    def test_returns_string(self) -> None:
        from llm_judge_rubric import build_rubric_prompt

        result = build_rubric_prompt(SAMPLE_PROPOSAL, variant="detailed")
        assert isinstance(result, str)

    def test_contains_claim(self) -> None:
        from llm_judge_rubric import build_rubric_prompt

        result = build_rubric_prompt(SAMPLE_PROPOSAL, variant="detailed")
        assert "Stellar nucleosynthesis" in result

    def test_contains_four_dimensions(self) -> None:
        from llm_judge_rubric import build_rubric_prompt

        result = build_rubric_prompt(SAMPLE_PROPOSAL, variant="detailed")
        for dim in ["plausibility", "novelty", "falsifiability", "clarity"]:
            assert dim in result.lower()

    def test_variant_concise_different_from_detailed(self) -> None:
        from llm_judge_rubric import build_rubric_prompt

        detailed = build_rubric_prompt(SAMPLE_PROPOSAL, variant="detailed")
        concise = build_rubric_prompt(SAMPLE_PROPOSAL, variant="concise")
        assert detailed != concise

    def test_invalid_variant_raises(self) -> None:
        from llm_judge_rubric import build_rubric_prompt

        with pytest.raises(ValueError):
            build_rubric_prompt(SAMPLE_PROPOSAL, variant="unknown_variant")

    def test_instructs_json_output(self) -> None:
        from llm_judge_rubric import build_rubric_prompt

        result = build_rubric_prompt(SAMPLE_PROPOSAL, variant="concise")
        assert "json" in result.lower() or "JSON" in result


# ---------------------------------------------------------------------------
# parse_rubric_scores
# ---------------------------------------------------------------------------


class TestParseRubricScores:
    def _mock_response(self, scores: dict) -> str:
        return json.dumps(scores)

    def test_parses_valid_scores(self) -> None:
        from llm_judge_rubric import parse_rubric_scores

        response = self._mock_response(
            {"plausibility": 4, "novelty": 3, "falsifiability": 5, "clarity": 4}
        )
        result = parse_rubric_scores(response)
        assert result["plausibility"] == 4
        assert result["novelty"] == 3
        assert result["falsifiability"] == 5
        assert result["clarity"] == 4

    def test_rejects_out_of_range_scores(self) -> None:
        from llm_judge_rubric import parse_rubric_scores

        response = self._mock_response(
            {"plausibility": 6, "novelty": 3, "falsifiability": 5, "clarity": 4}
        )
        with pytest.raises(ValueError):
            parse_rubric_scores(response)

    def test_rejects_score_below_1(self) -> None:
        from llm_judge_rubric import parse_rubric_scores

        response = self._mock_response(
            {"plausibility": 0, "novelty": 3, "falsifiability": 5, "clarity": 4}
        )
        with pytest.raises(ValueError):
            parse_rubric_scores(response)

    def test_rejects_missing_dimensions(self) -> None:
        from llm_judge_rubric import parse_rubric_scores

        response = self._mock_response({"plausibility": 4, "novelty": 3})
        with pytest.raises(ValueError):
            parse_rubric_scores(response)

    def test_handles_json_in_markdown_block(self) -> None:
        from llm_judge_rubric import parse_rubric_scores

        response = "```json\n" + json.dumps(
            {"plausibility": 4, "novelty": 3, "falsifiability": 5, "clarity": 4}
        ) + "\n```"
        result = parse_rubric_scores(response)
        assert result["plausibility"] == 4

    def test_handles_invalid_json_raises(self) -> None:
        from llm_judge_rubric import parse_rubric_scores

        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_rubric_scores("not valid json at all")


# ---------------------------------------------------------------------------
# compute_agreement
# ---------------------------------------------------------------------------


class TestComputeAgreement:
    def test_identical_scores_agreement_1(self) -> None:
        from llm_judge_rubric import compute_agreement

        s1 = {"plausibility": 4, "novelty": 3, "falsifiability": 5, "clarity": 4}
        s2 = {"plausibility": 4, "novelty": 3, "falsifiability": 5, "clarity": 4}
        assert compute_agreement(s1, s2) == pytest.approx(1.0)

    def test_completely_different_agreement_0(self) -> None:
        from llm_judge_rubric import compute_agreement

        s1 = {"plausibility": 1, "novelty": 1, "falsifiability": 1, "clarity": 1}
        s2 = {"plausibility": 5, "novelty": 5, "falsifiability": 5, "clarity": 5}
        assert compute_agreement(s1, s2) == pytest.approx(0.0)

    def test_partial_agreement(self) -> None:
        from llm_judge_rubric import compute_agreement

        # 2 matching, 2 within ±1, agreement should be > 0
        s1 = {"plausibility": 4, "novelty": 3, "falsifiability": 5, "clarity": 4}
        s2 = {"plausibility": 4, "novelty": 4, "falsifiability": 5, "clarity": 3}
        result = compute_agreement(s1, s2)
        assert 0.0 < result <= 1.0


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_single_entry(self) -> None:
        from llm_judge_rubric import aggregate_scores

        entries = [{"plausibility": 4.0, "novelty": 3.0, "falsifiability": 5.0, "clarity": 4.0}]
        result = aggregate_scores(entries)
        assert result["plausibility"] == pytest.approx(4.0)
        assert result["novelty"] == pytest.approx(3.0)

    def test_multiple_entries_mean(self) -> None:
        from llm_judge_rubric import aggregate_scores

        entries = [
            {"plausibility": 4.0, "novelty": 2.0, "falsifiability": 5.0, "clarity": 3.0},
            {"plausibility": 2.0, "novelty": 4.0, "falsifiability": 3.0, "clarity": 5.0},
        ]
        result = aggregate_scores(entries)
        assert result["plausibility"] == pytest.approx(3.0)
        assert result["novelty"] == pytest.approx(3.0)

    def test_empty_list_raises(self) -> None:
        from llm_judge_rubric import aggregate_scores

        with pytest.raises(ValueError):
            aggregate_scores([])
