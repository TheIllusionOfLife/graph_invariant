"""MLX LLM backend for Apple Silicon — singleton model cache + generate.

Lazy-imports mlx_lm so Linux/no-GPU users never pay the import cost.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from harmony.proposals.errors import LLMBackendError
from harmony.proposals.llm_proposer import _extract_proposal_dict

logger = logging.getLogger(__name__)

_MLX_CACHE: dict[str, tuple[Any, Any]] = {}


def _do_load_mlx_model(model_id: str) -> tuple[Any, Any]:
    """Actually load the model+tokenizer (separated for testability)."""
    try:
        from mlx_lm import load as mlx_load
    except ImportError as exc:
        raise ImportError(
            "mlx_lm is required for the MLX backend. "
            "Install with: uv pip install mlx-lm"
        ) from exc

    logger.info("Loading MLX model %s (this may take a moment)...", model_id)
    model, tokenizer = mlx_load(model_id, strict=False)
    logger.info("MLX model %s loaded.", model_id)
    return model, tokenizer


def _load_mlx_model(model_id: str) -> tuple[Any, Any]:
    """Load model+tokenizer with singleton caching."""
    if model_id not in _MLX_CACHE:
        _MLX_CACHE[model_id] = _do_load_mlx_model(model_id)
    return _MLX_CACHE[model_id]


def clear_mlx_cache() -> None:
    """Clear the singleton model cache (for test teardown)."""
    _MLX_CACHE.clear()


def mlx_generate(model: Any, tokenizer: Any, prompt: str, **kwargs: Any) -> str:
    """Thin wrapper around mlx_lm.generate (separated for testability)."""
    from mlx_lm import generate as _mlx_generate

    return _mlx_generate(model, tokenizer, prompt=prompt, **kwargs)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> prefix from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def generate_proposal_mlx(
    prompt: str,
    model_id: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Generate a proposal via mlx_lm.

    Returns dict with keys "response" (str) and "proposal_dict" (dict | None).
    Wraps runtime errors in LLMBackendError.
    """
    try:
        model, tokenizer = _load_mlx_model(model_id)

        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        raw = mlx_generate(
            model,
            tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        text = _strip_thinking(raw).strip()
        proposal_dict = _extract_proposal_dict(text)
        return {"response": text, "proposal_dict": proposal_dict}
    except LLMBackendError:
        raise
    except Exception as exc:
        raise LLMBackendError(str(exc)) from exc
