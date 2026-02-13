import pytest

from graph_invariant.llm_ollama import (
    IslandStrategy,
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
        "refine", ["def new_invariant(s):\n    return s['n']"], ["syntax error"], "diameter"
    )
    assert "Island mode: refine" in prompt
    assert "diameter" in prompt
    assert "Recent failures" in prompt


def test_build_prompt_uses_feature_dict_signature():
    prompt = build_prompt("free", [], [], "average_shortest_path_length")
    assert "new_invariant(s)" in prompt
    assert "new_invariant(G)" not in prompt


def test_build_prompt_lists_available_features():
    prompt = build_prompt("free", [], [], "diameter")
    for key in ("n", "m", "density", "degrees", "avg_degree", "avg_clustering"):
        assert key in prompt


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


def test_generate_candidate_code_respects_timeout(monkeypatch):
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "def new_invariant(G):\n    return 3"}

    captured: dict[str, object] = {}

    def fake_post(url, json, timeout, allow_redirects):  # noqa: ANN001
        captured["timeout"] = timeout
        del url, json, allow_redirects
        return DummyResponse()

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.post", fake_post)
    _ = generate_candidate_code(
        "p",
        "m",
        0.3,
        "http://localhost:11434/api/generate",
        timeout_sec=91.0,
    )
    assert captured["timeout"] == 91.0


def test_tags_endpoint_preserves_subpath():
    endpoint = _tags_endpoint("http://example.com/ollama/api/generate")
    assert endpoint == "http://example.com/ollama/api/tags"


def test_validate_ollama_url_rejects_non_local_hosts():
    with pytest.raises(ValueError):
        validate_ollama_url("http://169.254.169.254/api/generate", allow_remote=False)


def test_generate_candidate_code_retries_on_read_timeout(monkeypatch):
    import requests as req

    call_count = 0

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": "def new_invariant(G):\n    return 1"}

    def fake_post(url, json, timeout, allow_redirects):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise req.exceptions.ReadTimeout("timed out")
        return DummyResponse()

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.post", fake_post)
    code = generate_candidate_code("p", "m", 0.3, "http://localhost:11434/api/generate")
    assert "def new_invariant" in code
    assert call_count == 2


def test_generate_candidate_code_fails_after_max_retries(monkeypatch):
    import requests as req

    call_count = 0

    def fake_post(url, json, timeout, allow_redirects):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        raise req.exceptions.ReadTimeout("timed out")

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.post", fake_post)
    with pytest.raises(req.exceptions.ReadTimeout):
        generate_candidate_code("p", "m", 0.3, "http://localhost:11434/api/generate")
    assert call_count == 3


def test_generate_candidate_code_no_retry_on_connection_error(monkeypatch):
    import requests as req

    call_count = 0

    def fake_post(url, json, timeout, allow_redirects):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        raise req.exceptions.ConnectionError("refused")

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.post", fake_post)
    with pytest.raises(req.exceptions.ConnectionError):
        generate_candidate_code("p", "m", 0.3, "http://localhost:11434/api/generate")
    assert call_count == 1


def test_build_prompt_refinement_strategy():
    prompt = build_prompt(
        "free",
        ["def new_invariant(s):\n    return s['n']"],
        [],
        "diameter",
        strategy=IslandStrategy.REFINEMENT,
    )
    assert any(word in prompt.lower() for word in ("improve", "refine"))


def test_build_prompt_combination_strategy():
    prompt = build_prompt(
        "free",
        ["def new_invariant(s):\n    return s['n']", "def new_invariant(s):\n    return s['m']"],
        [],
        "diameter",
        strategy=IslandStrategy.COMBINATION,
    )
    assert "combine" in prompt.lower()


def test_build_prompt_novel_strategy():
    prompt = build_prompt(
        "free",
        [],
        [],
        "diameter",
        strategy=IslandStrategy.NOVEL,
    )
    assert "novel" in prompt.lower()


def test_build_prompt_includes_anti_patterns():
    for strategy in IslandStrategy:
        prompt = build_prompt("free", [], [], "diameter", strategy=strategy)
        assert "FORBIDDEN" in prompt
        assert "BFS" in prompt or "bfs" in prompt.lower()


def test_build_prompt_includes_formula_examples():
    for strategy in IslandStrategy:
        prompt = build_prompt("free", [], [], "diameter", strategy=strategy)
        assert "def new_invariant(s)" in prompt
        assert "s['n']" in prompt or "s['m']" in prompt


def test_build_prompt_feature_dict_signature_with_strategy():
    prompt = build_prompt("free", [], [], "diameter", strategy=IslandStrategy.REFINEMENT)
    assert "new_invariant(s)" in prompt
    assert "new_invariant(G)" not in prompt
    assert "pre-computed" in prompt.lower() or "feature" in prompt.lower()


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
