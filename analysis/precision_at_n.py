"""Precision@N against withheld backtesting edges.

Addresses reviewer A-Q2: quantitative evaluation of proposal accuracy
against ground-truth edges held out from the KG.
"""

from __future__ import annotations


def precision_at_n(
    proposals_ranked: list[tuple[str, str, str]],
    withheld: list[tuple[str, str, str]],
    n: int,
) -> float:
    """Compute Precision@N for ranked proposals against withheld edges.

    Args:
        proposals_ranked: Ranked list of (source, target, edge_type) triples.
        withheld: Ground-truth (source, target, edge_type) triples.
        n: Number of top proposals to evaluate.

    Returns:
        Fraction of top-N proposals that match a withheld edge.
    """
    if not proposals_ranked or n <= 0:
        return 0.0

    withheld_set = {(s, t, et) for s, t, et in withheld}
    top_n = proposals_ranked[:n]
    hits = sum(1 for triple in top_n if triple in withheld_set)
    return float(hits) / len(top_n)
