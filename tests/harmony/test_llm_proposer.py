"""Tests for harmony.proposals.llm_proposer — prompt building, JSON parsing, HTTP mocking.

TDD: all tests are written BEFORE the implementation. They verify:
  - island_strategy() cycles through REFINEMENT → COMBINATION → REFINEMENT → NOVEL
  - build_proposal_prompt() includes KG stats and strategy preamble
  - build_proposal_prompt() with constrained=True lists all entity IDs and EdgeType names
  - _extract_proposal_dict() parses fenced and raw JSON; returns None on garbage
  - generate_proposal_payload() retries on network errors and returns parsed dict
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.proposals.llm_proposer import (
    ProposalStrategy,
    _extract_proposal_dict,
    build_proposal_prompt,
    generate_proposal_payload,
    island_strategy,
)
from harmony.types import EdgeType

# ---------------------------------------------------------------------------
# island_strategy
# ---------------------------------------------------------------------------


class TestIslandStrategy:
    def test_island_strategy_cycles_correctly(self):
        """Islands 0..3 → REFINEMENT, COMBINATION, REFINEMENT, NOVEL."""
        assert island_strategy(0) == ProposalStrategy.REFINEMENT
        assert island_strategy(1) == ProposalStrategy.COMBINATION
        assert island_strategy(2) == ProposalStrategy.REFINEMENT
        assert island_strategy(3) == ProposalStrategy.NOVEL

    def test_island_strategy_wraps_at_4(self):
        """Island 4 should wrap back to REFINEMENT (same as island 0)."""
        assert island_strategy(4) == island_strategy(0)
        assert island_strategy(5) == island_strategy(1)

    def test_island_strategy_is_enum(self):
        for i in range(8):
            assert isinstance(island_strategy(i), ProposalStrategy)


# ---------------------------------------------------------------------------
# build_proposal_prompt
# ---------------------------------------------------------------------------


class TestBuildProposalPrompt:
    @pytest.fixture()
    def kg(self):
        return build_linear_algebra_kg()

    def test_includes_entity_count(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [])
        assert str(kg.num_entities) in prompt

    def test_includes_edge_count(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.NOVEL, [], [])
        assert str(kg.num_edges) in prompt

    def test_includes_domain(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [])
        assert kg.domain in prompt

    def test_refinement_preamble(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [])
        assert "refine" in prompt.lower() or "improve" in prompt.lower()

    def test_combination_preamble(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.COMBINATION, [], [])
        assert "combine" in prompt.lower() or "merge" in prompt.lower()

    def test_novel_preamble(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.NOVEL, [], [])
        assert "novel" in prompt.lower() or "invent" in prompt.lower()

    def test_top_proposals_included(self, kg):
        top = ['{"claim": "Eigenvectors depend on determinant."}']
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, top, [])
        assert "Eigenvectors depend on determinant" in prompt

    def test_recent_failures_included(self, kg):
        failures = ["'claim' must be at least 10 characters"]
        prompt = build_proposal_prompt(kg, ProposalStrategy.NOVEL, [], failures)
        assert "claim" in prompt

    def test_requests_json_output(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [])
        # Must ask for JSON output
        assert "json" in prompt.lower() or "JSON" in prompt

    def test_constrained_lists_all_entity_ids(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [], constrained=True)
        for entity_id in kg.entities:
            assert entity_id in prompt

    def test_constrained_lists_all_edge_type_names(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [], constrained=True)
        for et in EdgeType:
            assert et.name in prompt

    def test_free_mode_includes_entity_sample(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [], constrained=False)
        # Free mode should include some entity IDs so the LLM knows what to reference
        entity_ids = list(kg.entities.keys())
        present = sum(1 for eid in entity_ids if eid in prompt)
        # At least some entities should appear (sample or full list for small KGs)
        assert present >= 1

    def test_free_mode_includes_edge_types(self, kg):
        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [], constrained=False)
        # Free mode should also list valid edge types
        for et in EdgeType:
            assert et.name in prompt

    def test_free_mode_entity_sample_capped(self):
        """For KGs with >20 entities, free mode caps at _MAX_FREE_ENTITY_SAMPLE."""
        from harmony.proposals.llm_proposer import _MAX_FREE_ENTITY_SAMPLE

        kg = build_linear_algebra_kg()
        assert kg.num_entities > _MAX_FREE_ENTITY_SAMPLE, "Need >20 entities for cap test"

        prompt = build_proposal_prompt(kg, ProposalStrategy.REFINEMENT, [], [], constrained=False)

        # Should show the "(showing N of M)" suffix
        assert f"(showing {_MAX_FREE_ENTITY_SAMPLE} of {kg.num_entities})" in prompt

        # Extract the EXAMPLE ENTITY line and count entity IDs listed
        for line in prompt.splitlines():
            if line.startswith("EXAMPLE ENTITY"):
                # Entity IDs are comma-separated after the colon
                _, _, entity_csv = line.partition(": ")
                listed_ids = [eid.strip() for eid in entity_csv.split(",") if eid.strip()]
                assert len(listed_ids) == _MAX_FREE_ENTITY_SAMPLE
                # All listed IDs must be real KG entities
                for eid in listed_ids:
                    assert eid in kg.entities, f"Listed entity '{eid}' not in KG"
                break
        else:
            pytest.fail("No 'EXAMPLE ENTITY' line found in prompt")


# ---------------------------------------------------------------------------
# _extract_proposal_dict
# ---------------------------------------------------------------------------


class TestExtractProposalDict:
    def test_parses_fenced_json_block(self):
        text = '```json\n{"id": "p1", "proposal_type": "add_edge"}\n```'
        result = _extract_proposal_dict(text)
        assert result is not None
        assert result["id"] == "p1"

    def test_parses_raw_json_no_fence(self):
        text = '{"id": "p2", "claim": "Something happens."}'
        result = _extract_proposal_dict(text)
        assert result is not None
        assert result["id"] == "p2"

    def test_returns_none_on_garbage(self):
        result = _extract_proposal_dict("This is not JSON at all!")
        assert result is None

    def test_returns_none_on_empty_string(self):
        assert _extract_proposal_dict("") is None

    def test_returns_none_on_non_object_json(self):
        # JSON arrays are not a valid Proposal dict
        result = _extract_proposal_dict("[1, 2, 3]")
        assert result is None

    def test_extracts_first_json_from_mixed_text(self):
        text = 'Here is the proposal:\n{"id": "p3", "type": "add"}\nsome extra text'
        result = _extract_proposal_dict(text)
        assert result is not None
        assert result["id"] == "p3"

    def test_handles_nested_json(self):
        nested = {"id": "p4", "nested": {"key": "value"}}
        text = json.dumps(nested)
        result = _extract_proposal_dict(text)
        assert result is not None
        assert result["nested"]["key"] == "value"


# ---------------------------------------------------------------------------
# generate_proposal_payload (HTTP mocked)
# ---------------------------------------------------------------------------


class TestGenerateProposalPayload:
    def _mock_response(self, text: str) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {"response": text}
        resp.raise_for_status.return_value = None
        return resp

    def test_returns_response_and_proposal_dict_on_valid_json(self):
        json_text = '{"id": "test", "proposal_type": "add_edge", "claim": "test claim"}'
        with (
            patch("harmony.proposals.llm_proposer.validate_ollama_url") as mock_url,
            patch("requests.post") as mock_post,
        ):
            mock_url.return_value = ("http://127.0.0.1:11434/api/generate", {})
            mock_post.return_value = self._mock_response(json_text)
            result = generate_proposal_payload(
                prompt="test prompt",
                model="mistral",
                temperature=0.3,
                url="http://localhost:11434/api/generate",
            )
        assert "response" in result
        assert "proposal_dict" in result
        assert result["proposal_dict"] is not None
        assert result["proposal_dict"]["id"] == "test"

    def test_proposal_dict_is_none_on_garbage_response(self):
        with (
            patch("harmony.proposals.llm_proposer.validate_ollama_url") as mock_url,
            patch("requests.post") as mock_post,
        ):
            mock_url.return_value = ("http://127.0.0.1:11434/api/generate", {})
            mock_post.return_value = self._mock_response("Not valid JSON at all!")
            result = generate_proposal_payload(
                prompt="test",
                model="mistral",
                temperature=0.3,
                url="http://localhost:11434/api/generate",
            )
        assert result["proposal_dict"] is None

    def test_retries_on_connection_error(self):
        import requests as req_module

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise req_module.exceptions.ConnectionError("refused")
            resp = self._mock_response('{"id": "ok"}')
            return resp

        with (
            patch("harmony.proposals.llm_proposer.validate_ollama_url") as mock_url,
            patch("requests.post", side_effect=side_effect),
            patch("time.sleep"),  # skip actual sleep
        ):
            mock_url.return_value = ("http://127.0.0.1:11434/api/generate", {})
            result = generate_proposal_payload(
                prompt="test",
                model="mistral",
                temperature=0.3,
                url="http://localhost:11434/api/generate",
                max_retries=3,
            )
        assert call_count == 3
        assert result["proposal_dict"] is not None

    def test_raises_after_max_retries_exhausted(self):
        import requests as req_module

        with (
            patch("harmony.proposals.llm_proposer.validate_ollama_url") as mock_url,
            patch("requests.post", side_effect=req_module.exceptions.ConnectionError("refused")),
            patch("time.sleep"),
        ):
            mock_url.return_value = ("http://127.0.0.1:11434/api/generate", {})
            with pytest.raises(req_module.exceptions.ConnectionError):
                generate_proposal_payload(
                    prompt="test",
                    model="mistral",
                    temperature=0.3,
                    url="http://localhost:11434/api/generate",
                    max_retries=2,
                )
