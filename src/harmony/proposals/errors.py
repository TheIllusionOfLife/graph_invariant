"""Shared error types for LLM backends (Ollama / MLX)."""


class LLMBackendError(Exception):
    """Raised by LLM backends (Ollama/MLX) on runtime failures."""
