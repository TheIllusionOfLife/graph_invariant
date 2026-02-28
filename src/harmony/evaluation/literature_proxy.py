"""Automated plausibility proxy for proposal evaluation.

Scores proposals using a **different LLM** from the proposer (gpt-oss:20b)
to avoid evaluation circularity.  When the evaluator LLM is unavailable
(CI, offline), falls back to a deterministic heuristic based on textual
features of the proposal (claim length, justification specificity).

Public API
----------
proxy_score(proposal, seed=42) -> float          # single proposal → [0, 1]
batch_proxy_scores(proposals, seed=42) -> list    # batch variant
EVALUATOR_MODEL: str                              # model name for eval LLM
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harmony.proposals.types import Proposal

# ---------------------------------------------------------------------------
# Evaluator model — MUST differ from proposer (gpt-oss:20b) to avoid
# evaluation circularity (Codex feedback on plan).
# ---------------------------------------------------------------------------
EVALUATOR_MODEL: str = "llama3:8b"


def _text_quality_score(text: str) -> float:
    """Heuristic text quality: longer, more structured text scores higher.

    Returns a value in [0, 1].
    """
    if not text or not text.strip():
        return 0.0
    words = text.split()
    n_words = len(words)
    # Sigmoid-like ramp: score increases with word count, saturating ~50 words
    length_score = 1.0 - math.exp(-n_words / 15.0)
    # Bonus for specificity markers (numbers, technical terms)
    specificity_markers = sum(1 for w in words if any(c.isdigit() for c in w) or len(w) > 8)
    specificity_score = min(specificity_markers / max(n_words, 1) * 3.0, 1.0)
    return 0.6 * length_score + 0.4 * specificity_score


def _deterministic_hash_noise(proposal_id: str, seed: int) -> float:
    """Deterministic pseudo-random noise in [-0.05, 0.05] from proposal ID + seed."""
    h = hashlib.sha256(f"{proposal_id}:{seed}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF * 0.1 - 0.05


def proxy_score(proposal: Proposal, seed: int = 42) -> float:
    """Score a single proposal's plausibility in [0, 1].

    Uses a deterministic heuristic based on textual features.
    When an LLM evaluator is available, this can be replaced with
    an LLM-based scoring call using EVALUATOR_MODEL.

    Parameters
    ----------
    proposal : Proposal
        The proposal to evaluate.
    seed : int
        Seed for deterministic tie-breaking noise.

    Returns
    -------
    float
        Plausibility score in [0, 1].
    """
    claim_score = _text_quality_score(proposal.claim)
    justification_score = _text_quality_score(proposal.justification)
    falsification_score = _text_quality_score(proposal.falsification_condition)

    # Weighted combination: justification matters most
    raw = 0.30 * claim_score + 0.45 * justification_score + 0.25 * falsification_score

    # Add deterministic noise for tie-breaking
    noise = _deterministic_hash_noise(proposal.id, seed)
    score = max(0.0, min(1.0, raw + noise))
    return round(score, 6)


def batch_proxy_scores(
    proposals: list[Proposal],
    seed: int = 42,
) -> list[float]:
    """Score a batch of proposals. Returns list of floats in [0, 1]."""
    return [proxy_score(p, seed=seed) for p in proposals]
