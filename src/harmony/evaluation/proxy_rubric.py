"""Automated 5-criterion rubric for proposal quality evaluation.

Each proposal is scored on five dimensions (1–5 scale):
  novelty       — Does the claim go beyond restating known facts?
  plausibility  — Is the claim physically/scientifically reasonable?
  specificity   — Does the claim make concrete, testable predictions?
  testability   — Is the falsification condition experimentally feasible?
  relevance     — Does the claim relate meaningfully to the KG domain?

When an LLM evaluator is available, scores come from structured prompting.
Offline/CI mode uses deterministic heuristics based on textual features.

Public API
----------
rubric_score(proposal, seed=42) -> dict[str, float]   # 5 criterion scores
rubric_aggregate(scores) -> float                      # mean of 5 scores
batch_rubric_scores(proposals, seed=42) -> list        # batch variant
RUBRIC_CRITERIA: dict[str, str]                        # criterion definitions
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harmony.proposals.types import Proposal

# ---------------------------------------------------------------------------
# Rubric criterion definitions
# ---------------------------------------------------------------------------
RUBRIC_CRITERIA: dict[str, str] = {
    "novelty": (
        "Does the proposal go beyond restating well-known facts? "
        "Score 1 for trivial restatements, 5 for genuinely novel hypotheses."
    ),
    "plausibility": (
        "Is the claimed relationship scientifically reasonable? "
        "Score 1 for contradicting established knowledge, 5 for well-grounded claims."
    ),
    "specificity": (
        "Does the claim make concrete, testable predictions rather than vague assertions? "
        "Score 1 for hand-wavy generalities, 5 for precise quantitative predictions."
    ),
    "testability": (
        "Is the falsification condition experimentally or observationally feasible? "
        "Score 1 for untestable conditions, 5 for clearly actionable experiments."
    ),
    "relevance": (
        "Does the claim meaningfully relate to the target KG domain? "
        "Score 1 for off-topic claims, 5 for claims central to domain advancement."
    ),
}


# ---------------------------------------------------------------------------
# Deterministic heuristic scoring (offline/CI mode)
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _specificity_ratio(text: str) -> float:
    """Fraction of words that are 'specific' (long or contain digits)."""
    words = text.split()
    if not words:
        return 0.0
    specific = sum(1 for w in words if len(w) > 7 or any(c.isdigit() for c in w))
    return specific / len(words)


def _hash_deterministic(key: str, seed: int, low: float, high: float) -> float:
    """Deterministic float in [low, high] from hashed key+seed."""
    h = hashlib.sha256(f"{key}:{seed}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return low + frac * (high - low)


def _score_criterion(
    text: str,
    criterion: str,
    proposal_id: str,
    seed: int,
) -> float:
    """Score a single criterion on [1, 5] using textual heuristics."""
    wc = _word_count(text)
    spec = _specificity_ratio(text)

    # Length-based component (more words → higher base score)
    length_base = 1.0 + 3.0 * (1.0 - math.exp(-wc / 12.0))

    # Specificity bonus
    spec_bonus = spec * 1.5

    raw = length_base + spec_bonus

    # Add small deterministic noise per criterion
    noise = _hash_deterministic(f"{proposal_id}:{criterion}", seed, -0.3, 0.3)
    return max(1.0, min(5.0, round(raw + noise, 2)))


def rubric_score(proposal: Proposal, seed: int = 42) -> dict[str, float]:
    """Score a proposal on all 5 rubric criteria.

    Parameters
    ----------
    proposal : Proposal
        The proposal to evaluate.
    seed : int
        Seed for deterministic scoring.

    Returns
    -------
    dict[str, float]
        Criterion name → score in [1, 5].
    """
    # Map criteria to their most relevant text field
    text_map = {
        "novelty": proposal.claim,
        "plausibility": proposal.justification,
        "specificity": proposal.claim,
        "testability": proposal.falsification_condition,
        "relevance": proposal.justification,
    }

    return {
        criterion: _score_criterion(text_map[criterion], criterion, proposal.id, seed)
        for criterion in RUBRIC_CRITERIA
    }


def rubric_aggregate(scores: dict[str, float]) -> float:
    """Compute the mean of rubric criterion scores.

    Parameters
    ----------
    scores : dict[str, float]
        Output from rubric_score().

    Returns
    -------
    float
        Mean score in [1, 5].
    """
    values = list(scores.values())
    return sum(values) / len(values)


def batch_rubric_scores(
    proposals: list[Proposal],
    seed: int = 42,
) -> list[dict[str, float]]:
    """Score a batch of proposals on all rubric criteria."""
    return [rubric_score(p, seed=seed) for p in proposals]
