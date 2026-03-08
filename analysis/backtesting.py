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


@dataclass(slots=True)
class SoftBacktestResult:
    """Result of soft (typed-neighbourhood) backtesting."""

    n_proposals: int
    n_hidden: int
    soft_precision_at_5: float
    soft_precision_at_10: float
    soft_precision_at_20: float
    soft_recall_at_5: float
    soft_recall_at_10: float
    soft_recall_at_20: float


def _soft_matches(
    proposal: tuple[str, str, str],
    hidden_set_source_type: set[tuple[str, str]],
    hidden_set_type_target: set[tuple[str, str]],
) -> bool:
    """True if proposal shares (source, edge_type) or (edge_type, target) with any hidden edge."""
    src, tgt, et = proposal
    return (src, et) in hidden_set_source_type or (et, tgt) in hidden_set_type_target


def _soft_precision_at_n(
    proposals: list[tuple[str, str, str]],
    hidden_source_type: set[tuple[str, str]],
    hidden_type_target: set[tuple[str, str]],
    n: int,
) -> float:
    if not proposals or n <= 0:
        return 0.0
    seen: set[tuple[str, str, str]] = set()
    unique_top: list[tuple[str, str, str]] = []
    for p in proposals:
        if p not in seen:
            seen.add(p)
            unique_top.append(p)
            if len(unique_top) == n:
                break
    if not unique_top:
        return 0.0
    hits = sum(1 for p in unique_top if _soft_matches(p, hidden_source_type, hidden_type_target))
    return float(hits) / len(unique_top)


def _soft_recall_at_n(
    proposals: list[tuple[str, str, str]],
    hidden_edges: list[tuple[str, str, str]],
    hidden_source_type: set[tuple[str, str]],
    hidden_type_target: set[tuple[str, str]],
    n: int,
) -> float:
    if not hidden_edges or not proposals or n <= 0:
        return 0.0
    seen: set[tuple[str, str, str]] = set()
    unique_top: list[tuple[str, str, str]] = []
    for p in proposals:
        if p not in seen:
            seen.add(p)
            unique_top.append(p)
            if len(unique_top) == n:
                break
    # For each hidden edge, check if any top-N proposal soft-matches it
    covered = 0
    for s, t, et in hidden_edges:
        for p_src, p_tgt, p_et in unique_top:
            if (p_src == s and p_et == et) or (p_et == et and p_tgt == t):
                covered += 1
                break
    return float(covered) / len(hidden_edges)


def soft_backtest_proposals(
    proposals_ranked: list[tuple[str, str, str]],
    hidden_edges: list[tuple[str, str, str]],
) -> SoftBacktestResult:
    """Typed-neighbourhood soft backtesting against hidden edges.

    A proposal (src, tgt, et) soft-matches a hidden edge (s, t, e) if:
      - src == s AND et == e  (source-type match), OR
      - et == e AND tgt == t  (type-target match).

    This tests whether proposals target structurally relevant neighbourhoods
    even when exact triples don't match.

    Parameters
    ----------
    proposals_ranked:
        Ranked list of (source, target, edge_type) triples from the archive.
    hidden_edges:
        Ground-truth (source, target, edge_type) triples withheld during construction.

    Returns
    -------
    SoftBacktestResult with soft_precision@5/10/20 and soft_recall@5/10/20.
    """
    hidden_source_type: set[tuple[str, str]] = {(s, et) for s, t, et in hidden_edges}
    hidden_type_target: set[tuple[str, str]] = {(et, t) for s, t, et in hidden_edges}

    return SoftBacktestResult(
        n_proposals=len(proposals_ranked),
        n_hidden=len(hidden_edges),
        soft_precision_at_5=_soft_precision_at_n(proposals_ranked, hidden_source_type, hidden_type_target, 5),
        soft_precision_at_10=_soft_precision_at_n(proposals_ranked, hidden_source_type, hidden_type_target, 10),
        soft_precision_at_20=_soft_precision_at_n(proposals_ranked, hidden_source_type, hidden_type_target, 20),
        soft_recall_at_5=_soft_recall_at_n(proposals_ranked, hidden_edges, hidden_source_type, hidden_type_target, 5),
        soft_recall_at_10=_soft_recall_at_n(proposals_ranked, hidden_edges, hidden_source_type, hidden_type_target, 10),
        soft_recall_at_20=_soft_recall_at_n(proposals_ranked, hidden_edges, hidden_source_type, hidden_type_target, 20),
    )
