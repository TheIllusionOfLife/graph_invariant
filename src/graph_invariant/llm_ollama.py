import ipaddress
import re
import time
from enum import StrEnum
from urllib.parse import urlparse

import requests


class IslandStrategy(StrEnum):
    """Prompt strategy assigned to each island in the evolutionary search."""

    REFINEMENT = "refinement"
    COMBINATION = "combination"
    NOVEL = "novel"


_BASE_FEATURE_KEYS_DOC = (
    "Available keys in s: n (node count), m (edge count), density, avg_degree, "
    "max_degree, min_degree, std_degree, avg_clustering, transitivity, "
    "degree_assortativity, num_triangles, degrees (sorted degree list)"
)

_SPECTRAL_FEATURE_KEYS_DOC = (
    ", "
    "laplacian_lambda2, laplacian_lambda_max, laplacian_spectral_gap, "
    "normalized_laplacian_lambda2, laplacian_energy_ratio."
)

_ANTI_PATTERNS = (
    "\nFORBIDDEN: Do NOT return a single feature directly (e.g., return s['diameter']).\n"
    "Do NOT implement BFS/DFS/shortest-path algorithms.\n"
    "The input is a pre-computed feature dict, not a graph object.\n"
)

_FORMULA_EXAMPLES = (
    "\nExample formulas:\n"
    "def new_invariant(s): return s['n'] / (s['m'] + 1)\n"
    "def new_invariant(s): return np.log(s['n']) * np.mean(s['degrees'])"
    " / max(s['max_degree'], 1)\n"
    "def new_invariant(s): return np.sqrt(s['n']) / (1 + s['avg_clustering'])\n"
)

_TARGET_CONTEXT: dict[str, str] = {
    "average_shortest_path_length": (
        "Think about how it relates to density, degree distribution, "
        "clustering, and other structural features."
    ),
    "algebraic_connectivity": (
        "The Fiedler value is the 2nd-smallest Laplacian eigenvalue. "
        "Think about degree distribution, connectivity patterns, spectral relationships."
    ),
}

_STRATEGY_INSTRUCTIONS_CORRELATION: dict[IslandStrategy, str] = {
    IslandStrategy.REFINEMENT: (
        "Improve the best existing formula. Make small targeted changes "
        "to refine accuracy while keeping the formula simple."
    ),
    IslandStrategy.COMBINATION: (
        "Combine elements from the top 2 formulas into a new one. "
        "Merge their strengths into a single improved formula."
    ),
    IslandStrategy.NOVEL: "Invent a completely novel mathematical formula. ",
}

_STRATEGY_INSTRUCTIONS_BOUNDS: dict[IslandStrategy, str] = {
    IslandStrategy.REFINEMENT: "Tighten the best bound. Reduce the gap while keeping it valid.",
    IslandStrategy.COMBINATION: (
        "Combine elements from the top 2 bounds into a new one. "
        "Merge their strengths into a single tighter bound."
    ),
    IslandStrategy.NOVEL: "Invent a novel inequality. ",
}

_BOUNDS_INSTRUCTIONS: dict[str, str] = {
    "upper_bound": (
        "\nFind f(x) such that f(x) >= y for ALL graphs. Tighter bounds score higher.\n"
        "Think about inequalities: AM-GM, degree-sum, spectral bounds.\n"
        "Do NOT return trivially large constants.\n"
    ),
    "lower_bound": (
        "\nFind f(x) such that f(x) <= y for ALL graphs. Tighter bounds score higher.\n"
        "Think about inequalities: AM-GM, degree-sum, spectral bounds.\n"
        "Do NOT return trivially small constants (like 0).\n"
    ),
}


def _sanitize_llm_output(text: str) -> str:
    """Strip code fences from LLM-generated text to prevent prompt injection."""
    return text.replace("```", "")


def build_prompt(
    island_mode: str,
    top_candidates: list[str],
    failures: list[str],
    target_name: str,
    strategy: IslandStrategy | None = None,
    fitness_mode: str = "correlation",
    include_spectral_feature_pack: bool = True,
) -> str:
    """Build an LLM prompt for candidate formula generation.

    When *strategy* is provided, the prompt includes strategy-specific
    instructions (refine / combine / novel), anti-pattern warnings, and
    example formulas.  When *fitness_mode* is ``upper_bound`` or
    ``lower_bound``, the prompt asks the LLM to produce an inequality
    rather than a correlation-maximizing formula.
    """
    top_block = (
        "\n\n".join(_sanitize_llm_output(c) for c in top_candidates[:3])
        if top_candidates
        else "None yet."
    )
    fail_block = "\n".join(_sanitize_llm_output(f) for f in failures[:3]) if failures else "None."

    is_bounds = fitness_mode in ("upper_bound", "lower_bound")
    strategy_table = (
        _STRATEGY_INSTRUCTIONS_BOUNDS if is_bounds else _STRATEGY_INSTRUCTIONS_CORRELATION
    )

    strategy_instruction = ""
    if strategy is not None:
        base = strategy_table[strategy]
        # Append target-specific context to the NOVEL strategy
        if strategy == IslandStrategy.NOVEL:
            target_ctx = _TARGET_CONTEXT.get(target_name, "")
            if target_ctx:
                base = base + target_ctx
        strategy_instruction = f"\nStrategy: {base}\n"

    target_context_block = ""
    target_ctx = _TARGET_CONTEXT.get(target_name)
    if target_ctx and strategy is None:
        target_context_block = f"\n{target_ctx}\n"

    bounds_block = ""
    if is_bounds:
        bounds_block = _BOUNDS_INSTRUCTIONS.get(fitness_mode, "")

    feature_keys_doc = _BASE_FEATURE_KEYS_DOC
    if include_spectral_feature_pack:
        feature_keys_doc = feature_keys_doc + _SPECTRAL_FEATURE_KEYS_DOC
    else:
        feature_keys_doc = feature_keys_doc + "."

    return (
        f"You are discovering graph invariant formulas for target `{target_name}`.\n"
        f"Island mode: {island_mode}\n"
        f"{strategy_instruction}"
        f"{target_context_block}"
        f"Best formulas:\n{top_block}\n"
        f"Recent failures:\n{fail_block}\n"
        "Return only python code defining `def new_invariant(s):` "
        "where s is a dict of pre-computed graph features.\n"
        f"{feature_keys_doc}\n"
        f"{_ANTI_PATTERNS}"
        f"{_FORMULA_EXAMPLES}"
        f"{bounds_block}"
    )


def _extract_code_block(text: str) -> str:
    match = re.search(r"```python\s+(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_candidate_code(
    prompt: str,
    model: str,
    temperature: float,
    url: str,
    allow_remote: bool = False,
    timeout_sec: float = 60.0,
) -> str:
    payload = generate_candidate_payload(
        prompt=prompt,
        model=model,
        temperature=temperature,
        url=url,
        allow_remote=allow_remote,
        timeout_sec=timeout_sec,
    )
    return payload["code"]


def generate_candidate_payload(
    prompt: str,
    model: str,
    temperature: float,
    url: str,
    allow_remote: bool = False,
    timeout_sec: float = 60.0,
    max_retries: int = 3,
) -> dict[str, str]:
    validate_ollama_url(url, allow_remote=allow_remote)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    last_exc: requests.exceptions.RequestException | None = None
    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            response = requests.post(url, json=payload, timeout=timeout_sec, allow_redirects=False)
            response.raise_for_status()
            body = response.json()
            text = str(body.get("response", "")).strip()
            return {"response": text, "code": _extract_code_block(text)}
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if isinstance(exc, requests.exceptions.HTTPError):
                status_code = exc.response.status_code if exc.response is not None else 0
                if 400 <= status_code < 500 and status_code != 429:
                    raise exc
            if attempt < attempts - 1:
                time.sleep(1.0)
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Max retries exceeded with no exception captured")


def _tags_endpoint(generate_url: str) -> str:
    parsed = urlparse(generate_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/generate"):
        path = path[: -len("/generate")] + "/tags"
    elif path:
        path = f"{path}/tags"
    else:
        path = "/api/tags"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def validate_ollama_url(generate_url: str, allow_remote: bool) -> None:
    parsed = urlparse(generate_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("ollama_url must use http or https")
    if not parsed.netloc:
        raise ValueError("ollama_url must include a host")
    if allow_remote:
        return

    host = parsed.hostname
    if host is None:
        raise ValueError("ollama_url must include a valid hostname")
    if host in {"localhost", "127.0.0.1", "::1"}:
        return
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            "ollama_url must target localhost unless allow_remote_ollama is true"
        ) from None
    if ip.is_loopback:
        return
    raise ValueError("ollama_url must target localhost unless allow_remote_ollama is true")


def list_available_models(generate_url: str, allow_remote: bool = False) -> list[str]:
    validate_ollama_url(generate_url, allow_remote=allow_remote)
    response = requests.get(_tags_endpoint(generate_url), timeout=30, allow_redirects=False)
    response.raise_for_status()
    body = response.json()
    models = body.get("models", [])
    if not isinstance(models, list):
        return []
    return [
        entry.get("name")
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]
