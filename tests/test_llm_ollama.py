from graph_invariant.llm_ollama import _extract_code_block, build_prompt, generate_candidate_code


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

    def fake_post(url, json, timeout):  # noqa: ANN001
        assert "model" in json
        assert timeout == 60
        assert isinstance(url, str)
        return DummyResponse()

    monkeypatch.setattr("graph_invariant.llm_ollama.requests.post", fake_post)
    code = generate_candidate_code("p", "m", 0.3, "http://localhost:11434/api/generate")
    assert "def new_invariant" in code
