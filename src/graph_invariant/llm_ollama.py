import ipaddress
import re
from enum import StrEnum
from urllib.parse import urlparse

import requests


class IslandStrategy(StrEnum):
    REFINEMENT = "refinement"
    COMBINATION = "combination"
    NOVEL = "novel"


_FEATURE_KEYS_DOC = (
    "Available keys in s: n (node count), m (edge count), density, avg_degree, "
    "max_degree, min_degree, std_degree, avg_clustering, transitivity, "
    "degree_assortativity, num_triangles, degrees (sorted degree list)."
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

_STRATEGY_INSTRUCTIONS: dict[IslandStrategy, str] = {
    IslandStrategy.REFINEMENT: (
        "Improve the best existing formula. Make small targeted changes "
        "to refine accuracy while keeping the formula simple."
    ),
    IslandStrategy.COMBINATION: (
        "Combine elements from the top 2 formulas into a new one. "
        "Merge their strengths into a single improved formula."
    ),
    IslandStrategy.NOVEL: (
        "Invent a completely novel mathematical formula. "
        "Think about how average path length relates to density, "
        "degree distribution, clustering, and other structural features."
    ),
}


def build_prompt(
    island_mode: str,
    top_candidates: list[str],
    failures: list[str],
    target_name: str,
    strategy: IslandStrategy | None = None,
) -> str:
    top_block = "\n\n".join(top_candidates[:3]) if top_candidates else "None yet."
    fail_block = "\n".join(failures[:3]) if failures else "None."

    strategy_instruction = ""
    if strategy is not None:
        strategy_instruction = f"\nStrategy: {_STRATEGY_INSTRUCTIONS[strategy]}\n"

    return (
        f"You are discovering graph invariant formulas for target `{target_name}`.\n"
        f"Island mode: {island_mode}\n"
        f"{strategy_instruction}"
        f"Best formulas:\n{top_block}\n"
        f"Recent failures:\n{fail_block}\n"
        "Return only python code defining `def new_invariant(s):` "
        "where s is a dict of pre-computed graph features.\n"
        f"{_FEATURE_KEYS_DOC}\n"
        f"{_ANTI_PATTERNS}"
        f"{_FORMULA_EXAMPLES}"
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
    last_exc: requests.exceptions.ReadTimeout | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout_sec, allow_redirects=False)
            response.raise_for_status()
            body = response.json()
            text = str(body.get("response", "")).strip()
            return {"response": text, "code": _extract_code_block(text)}
        except requests.exceptions.ReadTimeout as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                continue
    raise last_exc  # type: ignore[misc]


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
