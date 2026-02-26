"""LLM-based proposal generator for Harmony island search.

Provides:
  - ProposalStrategy enum and island_strategy() cycle
  - build_proposal_prompt() — strategy-aware KG-grounded prompt
  - _extract_proposal_dict() — JSON parser robust to LLM formatting quirks
  - generate_proposal_payload() — Ollama HTTP call with retry (mirrors llm_ollama.py)
"""

from __future__ import annotations

import json
import re
import time
from enum import StrEnum
from typing import Any

import requests

from graph_invariant.llm_ollama import validate_ollama_url
from harmony.types import EdgeType, KnowledgeGraph


class ProposalStrategy(StrEnum):
    REFINEMENT = "refinement"
    COMBINATION = "combination"
    NOVEL = "novel"


_STRATEGY_CYCLE_LIST: list[ProposalStrategy] = [
    ProposalStrategy.REFINEMENT,
    ProposalStrategy.COMBINATION,
    ProposalStrategy.REFINEMENT,
    ProposalStrategy.NOVEL,
]


def island_strategy(island_id: int) -> ProposalStrategy:
    """Return the ProposalStrategy assigned to *island_id* (wraps every 4)."""
    return _STRATEGY_CYCLE_LIST[island_id % len(_STRATEGY_CYCLE_LIST)]


_STRATEGY_PREAMBLE: dict[ProposalStrategy, str] = {
    ProposalStrategy.REFINEMENT: (
        "Strategy: REFINEMENT — improve the best existing proposal. "
        "Make targeted changes to refine the claim and strengthen the justification."
    ),
    ProposalStrategy.COMBINATION: (
        "Strategy: COMBINATION — merge the top-2 proposals into a single stronger one. "
        "Combine their strengths, eliminate redundancies."
    ),
    ProposalStrategy.NOVEL: (
        "Strategy: NOVEL — invent a completely new theoretical proposal. "
        "Do not copy existing proposals; explore unexplored relationships."
    ),
}

_MAX_FREE_ENTITY_SAMPLE = 20

_PROPOSAL_SCHEMA_HINT = """
Return ONLY a JSON object (no extra text) with these fields:
{
  "id": "<unique string>",
  "proposal_type": "<add_edge|remove_edge|add_entity|remove_entity>",
  "claim": "<1-sentence theoretical claim, ≥10 chars>",
  "justification": "<reasoning, ≥10 chars>",
  "falsification_condition": "<what would disprove it, ≥10 chars>",
  "kg_domain": "<domain name, ≥10 chars>",
  "source_entity": "<entity id or null>",
  "target_entity": "<entity id or null>",
  "edge_type": "<EdgeType name or null>",
  "entity_id": "<entity id or null>",
  "entity_type": "<entity type string or null>"
}
"""


def _sanitize_for_prompt(text: str) -> str:
    """Strip content that could hijack the system prompt (prompt injection defence)."""
    # Remove code fences that could break prompt structure
    sanitized = text.replace("```", "")
    # Truncate to avoid context overflow from adversarial prior LLM output
    return sanitized[:500]


def build_proposal_prompt(
    kg: KnowledgeGraph,
    strategy: ProposalStrategy,
    top_proposals: list[str],
    recent_failures: list[str],
    constrained: bool = False,
) -> str:
    """Build an LLM prompt for structured KG-mutation proposal generation.

    Parameters
    ----------
    kg:
        The knowledge graph being explored.
    strategy:
        Island strategy (refinement / combination / novel).
    top_proposals:
        JSON strings of the current best proposals (shown for context).
    recent_failures:
        Violation messages from recently rejected proposals.
    constrained:
        When True, enumerate all valid entity IDs and EdgeType names explicitly
        to guide a stagnated island back to producing valid proposals.
    """
    # KG stats block
    stats_block = (
        f"Knowledge Graph: domain='{kg.domain}', entities={kg.num_entities}, edges={kg.num_edges}"
    )

    # Strategy preamble
    strategy_block = _STRATEGY_PREAMBLE[strategy]

    # Top proposals block — sanitized to prevent prompt injection from prior LLM output
    if top_proposals:
        safe_top = [_sanitize_for_prompt(p) for p in top_proposals[:3]]
        top_block = "Top proposals so far:\n" + "\n---\n".join(safe_top)
    else:
        top_block = "Top proposals so far: None yet."

    # Recent failures block — sanitized
    if recent_failures:
        safe_failures = [_sanitize_for_prompt(f) for f in recent_failures[:5]]
        fail_block = "Recent validation failures (avoid these):\n" + "\n".join(safe_failures)
    else:
        fail_block = "Recent validation failures: None."

    # Entity + edge type enumeration block (always included for grounding)
    edge_type_list = ", ".join(et.name for et in EdgeType)
    sorted_entities = sorted(kg.entities.keys())
    if constrained:
        entity_list = ", ".join(sorted_entities)
        constrained_block = (
            f"\nVALID ENTITY IDs (use EXACTLY as written): {entity_list}"
            f"\nVALID EDGE TYPES (use EXACTLY as written): {edge_type_list}\n"
        )
    else:
        # Free mode: show a sample so the LLM knows the naming convention
        sample = sorted_entities[:_MAX_FREE_ENTITY_SAMPLE]
        if len(sorted_entities) > _MAX_FREE_ENTITY_SAMPLE:
            suffix = f" (showing {len(sample)} of {len(sorted_entities)})"
        else:
            suffix = ""
        entity_list = ", ".join(sample)
        constrained_block = (
            f"\nEXAMPLE ENTITY IDs from this KG{suffix}: {entity_list}"
            f"\nVALID EDGE TYPES: {edge_type_list}"
            f"\nIMPORTANT: source_entity and target_entity MUST be exact entity IDs from this KG.\n"
        )

    return (
        f"You are a theory-discovery agent for knowledge graph research.\n"
        f"{stats_block}\n\n"
        f"{strategy_block}\n\n"
        f"{top_block}\n\n"
        f"{fail_block}\n"
        f"{constrained_block}"
        f"{_PROPOSAL_SCHEMA_HINT}"
    )


def _extract_proposal_dict(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from LLM output text.

    Handles:
    - Fenced ` ```json ... ``` ` blocks
    - Raw JSON embedded in prose (scanned with json.JSONDecoder.raw_decode,
      which is string-aware and correctly handles ``{`` / ``}`` inside values)
    Returns None if no valid JSON object can be found.
    """
    if not text:
        return None

    # 1. Try fenced block: ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Progressive scan using raw_decode — correctly handles braces inside string values
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = start + 1

    return None


def generate_proposal_payload(
    prompt: str,
    model: str,
    temperature: float,
    url: str,
    allow_remote: bool = False,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call Ollama and return raw response text + parsed proposal dict.

    Returns
    -------
    dict with keys:
      - "response": raw text from LLM
      - "proposal_dict": parsed dict | None (None when JSON extraction failed)

    Mirrors ``generate_candidate_payload()`` from ``llm_ollama.py``:
    retries on transient errors, raises on 4xx (except 429).
    """
    safe_url, headers = validate_ollama_url(url, allow_remote=allow_remote)
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
            response = requests.post(
                safe_url,
                json=payload,
                headers=headers,
                timeout=timeout_sec,
                allow_redirects=False,
            )
            response.raise_for_status()
            body = response.json()
            text = str(body.get("response", "")).strip()
            proposal_dict = _extract_proposal_dict(text)
            return {"response": text, "proposal_dict": proposal_dict}
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
