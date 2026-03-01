"""Pairwise Spearman correlation of 4 Harmony components.

Addresses reviewer B-Q1: are the 4 components redundant or do they
capture distinct aspects of KG quality?
"""

from __future__ import annotations

import statistics

_COMPONENTS = ("compressibility", "coherence", "symmetry", "generativity")


def _rank(values: list[float]) -> list[float]:
    """Assign average ranks to values (1-based)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation between two lists.

    Returns NaN when n < 2 or either vector has zero variance
    (correlation is undefined for constant inputs).
    """
    n = len(x)
    if n < 2:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    mean_rx = statistics.mean(rx)
    mean_ry = statistics.mean(ry)
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def compute_correlation_matrix(
    scores: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute pairwise Spearman correlation matrix for 4 Harmony components.

    Args:
        scores: List of dicts, each with keys for all 4 components.

    Returns:
        Nested dict: matrix[comp_a][comp_b] = Spearman rho.
    """
    vectors = {comp: [s[comp] for s in scores] for comp in _COMPONENTS}
    matrix: dict[str, dict[str, float]] = {}
    for a in _COMPONENTS:
        matrix[a] = {}
        for b in _COMPONENTS:
            if a == b:
                matrix[a][b] = 1.0
            else:
                matrix[a][b] = _spearman(vectors[a], vectors[b])
    return matrix


def correlation_summary(scores: list[dict[str, float]]) -> str:
    """Return a formatted summary of the correlation matrix."""
    matrix = compute_correlation_matrix(scores)
    lines = [f"{'':>18} " + " ".join(f"{c:>15}" for c in _COMPONENTS)]
    for a in _COMPONENTS:
        vals = " ".join(f"{matrix[a][b]:>15.3f}" for b in _COMPONENTS)
        lines.append(f"{a:>18} {vals}")
    return "\n".join(lines)
