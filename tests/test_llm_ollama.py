import pytest

from graph_invariant.llm_ollama import (
    _extract_code_block,
    _tags_endpoint,
    build_prompt,
    generate_candidate_code,
    list_available_models,
    validate_ollama_url,
)


def test_extract_code_block_prefers_python_fence():
    text = "noise\n```python\ndef new_invariant(G):\n    return 1\n```\nmore"
    assert _extract_code_block(text) == "def new_invariant(G):\n    return 1"


def test_build_prompt_contains_key_sections():
    prompt = build_prompt(
        "refine", ["def new_invariant(G):\n    return 1"], ["syntax error"], "diameter"
    )
    assert "Island mode: refine" in prompt
    assert "diameter" in prompt
    assert "Recent failures" in prompt


def test_generate_candidate_code_parses_response(monkeypatch):
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "```python\ndef new_invariant(G):\n    return 7\n```"}

    def fake_post(url, json, timeout, allow_redirects):  # noqa: ANN001
        assert "model" in json
        assert timeout == 60
        assert isinstance(url, str)
        assert allow_redirects is False
        return DummyResponse()

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.post", fake_post)
    code = generate_candidate_code("p", "m", 0.3, "http://localhost:11434/api/generate")
    assert "def new_invariant" in code


def test_tags_endpoint_preserves_subpath():
    endpoint = _tags_endpoint("http://example.com/ollama/api/generate")
    assert endpoint == "http://example.com/ollama/api/tags"


def test_validate_ollama_url_rejects_non_local_hosts():
    with pytest.raises(ValueError):
        validate_ollama_url("http://169.254.169.254/api/generate", allow_remote=False)


def test_list_available_models_uses_no_redirects(monkeypatch):
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"models": [{"name": "gpt-oss:20b"}]}

    captured: dict[str, object] = {}

    def fake_get(url, timeout, allow_redirects):  # noqa: ANN001
        captured["url"] = url
        captured["timeout"] = timeout
        captured["allow_redirects"] = allow_redirects
        return DummyResponse()

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.get", fake_get)
    names = list_available_models("http://localhost:11434/api/generate")
    assert names == ["gpt-oss:20b"]
    assert captured["allow_redirects"] is False
