"""Backtesting: evaluate archive proposals against hidden (withheld) edges.

Provides external validity evidence by checking whether top-ranked
archive proposals match edges held out from the KG during construction.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.precision_at_n import precision_at_n, recall_at_n


@dataclass(slots=True)
class BacktestResult:
    """Result of backtesting proposals against hidden edges."""

    n_proposals: int
    n_hidden: int
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    matched_triples: list[tuple[str, str, str]] = field(default_factory=list)


def backtest_proposals(
    proposals_ranked: list[tuple[str, str, str]],
    hidden_edges: list[tuple[str, str, str]],
) -> BacktestResult:
    """Backtest ranked proposals against hidden (withheld) edges.

    Parameters
    ----------
    proposals_ranked:
        Ranked list of (source, target, edge_type) triples from archive.
    hidden_edges:
        Ground-truth (source, target, edge_type) triples withheld during
        KG construction.

    Returns
    -------
    BacktestResult with precision@5/10/20, recall@5/10/20, and matched triples.
    """
    hidden_set = {(s, t, et) for s, t, et in hidden_edges}

    # Find matched triples (in top proposals that match hidden edges)
    seen: set[tuple[str, str, str]] = set()
    matched: list[tuple[str, str, str]] = []
    for triple in proposals_ranked:
        if triple not in seen:
            seen.add(triple)
            if triple in hidden_set:
                matched.append(triple)

    return BacktestResult(
        n_proposals=len(proposals_ranked),
        n_hidden=len(hidden_set),
        precision_at_5=precision_at_n(proposals_ranked, hidden_edges, n=5),
        precision_at_10=precision_at_n(proposals_ranked, hidden_edges, n=10),
        precision_at_20=precision_at_n(proposals_ranked, hidden_edges, n=20),
        recall_at_5=recall_at_n(proposals_ranked, hidden_edges, n=5),
        recall_at_10=recall_at_n(proposals_ranked, hidden_edges, n=10),
        recall_at_20=recall_at_n(proposals_ranked, hidden_edges, n=20),
        matched_triples=matched,
    )
