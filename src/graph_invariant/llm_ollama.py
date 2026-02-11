import re

import requests


def build_prompt(
    island_mode: str,
    top_candidates: list[str],
    failures: list[str],
    target_name: str,
) -> str:
    top_block = "\n\n".join(top_candidates[:3]) if top_candidates else "None yet."
    fail_block = "\n".join(failures[:3]) if failures else "None."
    return (
        f"You are improving graph invariant formulas for target `{target_name}`.\n"
        f"Island mode: {island_mode}\n"
        f"Best formulas:\n{top_block}\n"
        f"Recent failures:\n{fail_block}\n"
        "Return only python code defining `def new_invariant(G):`."
    )


def _extract_code_block(text: str) -> str:
    match = re.search(r"```python\s+(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_candidate_code(prompt: str, model: str, temperature: float, url: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    text = str(body.get("response", "")).strip()
    return _extract_code_block(text)
