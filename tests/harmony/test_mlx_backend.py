"""Tests for harmony.proposals.mlx_backend — MLX LLM backend (all mocked).

TDD: written BEFORE implementation. Verifies:
  - generate_proposal_mlx() returns correct shape (response + proposal_dict keys)
  - Garbage LLM output → proposal_dict is None
  - Model loading is cached (singleton pattern)
  - clear_mlx_cache() empties the cache
  - <think>...</think> prefix is stripped before JSON extraction
  - Runtime errors are wrapped in LLMBackendError
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_mlx_cache():
    from harmony.proposals.mlx_backend import clear_mlx_cache

    clear_mlx_cache()
    yield
    clear_mlx_cache()


class TestGenerateProposalMlx:
    def test_returns_correct_shape(self):
        from harmony.proposals.mlx_backend import generate_proposal_mlx

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        valid_json = '{"id": "p1", "proposal_type": "add_edge", "claim": "test"}'

        with (
            patch(
                "harmony.proposals.mlx_backend._load_mlx_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "harmony.proposals.mlx_backend.mlx_generate",
                return_value=valid_json,
            ),
        ):
            result = generate_proposal_mlx(prompt="test", model_id="test-model")

        assert "response" in result
        assert "proposal_dict" in result
        assert result["proposal_dict"] is not None
        assert result["proposal_dict"]["id"] == "p1"

    def test_proposal_dict_none_on_garbage(self):
        from harmony.proposals.mlx_backend import generate_proposal_mlx

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        with (
            patch(
                "harmony.proposals.mlx_backend._load_mlx_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "harmony.proposals.mlx_backend.mlx_generate",
                return_value="This is not JSON at all!",
            ),
        ):
            result = generate_proposal_mlx(prompt="test", model_id="test-model")

        assert result["proposal_dict"] is None

    def test_strips_thinking_tokens(self):
        from harmony.proposals.mlx_backend import generate_proposal_mlx

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        # Response with <think>...</think> prefix before JSON
        thinking_response = (
            "<think>Let me analyze this...</think>"
            '{"id": "p2", "proposal_type": "add_edge", "claim": "stripped"}'
        )

        with (
            patch(
                "harmony.proposals.mlx_backend._load_mlx_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "harmony.proposals.mlx_backend.mlx_generate",
                return_value=thinking_response,
            ),
        ):
            result = generate_proposal_mlx(prompt="test", model_id="test-model")

        assert result["proposal_dict"] is not None
        assert result["proposal_dict"]["id"] == "p2"
        # The thinking prefix should be stripped from the response
        assert "<think>" not in result["response"]

    def test_wraps_errors_in_llm_backend_error(self):
        from harmony.proposals.errors import LLMBackendError
        from harmony.proposals.mlx_backend import generate_proposal_mlx

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        with (
            patch(
                "harmony.proposals.mlx_backend._load_mlx_model",
                return_value=(mock_model, mock_tokenizer),
            ),
            patch(
                "harmony.proposals.mlx_backend.mlx_generate",
                side_effect=RuntimeError("GPU OOM"),
            ),
        ):
            with pytest.raises(LLMBackendError, match="GPU OOM"):
                generate_proposal_mlx(prompt="test", model_id="test-model")


class TestMlxModelCache:
    def test_load_caches_model(self):
        from harmony.proposals.mlx_backend import _load_mlx_model, clear_mlx_cache

        clear_mlx_cache()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch(
            "harmony.proposals.mlx_backend._do_load_mlx_model",
            return_value=(mock_model, mock_tokenizer),
        ) as mock_load:
            result1 = _load_mlx_model("test-model")
            result2 = _load_mlx_model("test-model")

        # Should only load once (cached)
        assert mock_load.call_count == 1
        assert result1 is result2

    def test_clear_cache(self):
        from harmony.proposals.mlx_backend import _MLX_CACHE, clear_mlx_cache

        # Manually insert a cache entry
        _MLX_CACHE["fake-model"] = (MagicMock(), MagicMock())
        assert len(_MLX_CACHE) > 0

        clear_mlx_cache()
        assert len(_MLX_CACHE) == 0
