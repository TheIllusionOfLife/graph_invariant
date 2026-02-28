"""Assess testability of proposal falsification conditions.

Addresses reviewer C-Q5: what fraction of falsification conditions
are experimentally testable?
"""

from __future__ import annotations

import math

from harmony.proposals.types import Proposal

_SPECIFICITY_MARKERS = frozenset(
    {
        "measure",
        "experiment",
        "observe",
        "test",
        "detect",
        "yield",
        "produce",
        "show",
        "demonstrate",
        "falsif",
        "temperature",
        "pressure",
        "concentration",
        "frequency",
        "spectroscop",
        "diffraction",
        "microscop",
        "conductiv",
        "nm",
        "mk",
        "gpa",
        "ev",
        "hz",
    }
)


def _text_specificity(text: str) -> float:
    """Score text specificity based on length and scientific markers."""
    if not text.strip():
        return 0.0
    words = text.lower().split()
    n_words = len(words)
    length_score = 1.0 / (1.0 + math.exp(-0.15 * (n_words - 10)))
    marker_hits = sum(1 for w in words if any(m in w for m in _SPECIFICITY_MARKERS))
    marker_ratio = min(marker_hits / max(n_words, 1), 1.0)
    return 0.6 * length_score + 0.4 * marker_ratio


def falsification_score(proposal: Proposal) -> float:
    """Score the testability of a proposal's falsification condition.

    Returns a float in [0, 1] where higher means more testable.
    """
    return _text_specificity(proposal.falsification_condition)


def batch_falsification_scores(proposals: list[Proposal]) -> list[float]:
    """Score testability for a batch of proposals."""
    return [falsification_score(p) for p in proposals]


def testability_fraction(
    proposals: list[Proposal],
    threshold: float = 0.4,
) -> float:
    """Fraction of proposals with falsification score above threshold."""
    if not proposals:
        return 0.0
    scores = batch_falsification_scores(proposals)
    n_testable = sum(1 for s in scores if s >= threshold)
    return float(n_testable) / len(proposals)
