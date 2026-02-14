"""Target functions for graph invariant discovery.

Maps target names to safe evaluation functions that handle disconnected
graphs, convergence errors, and edge cases gracefully.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx


def _safe_average_shortest_path_length(graph: nx.Graph) -> float:
    """Return the average shortest path length, or 0.0 for empty/disconnected graphs."""
    if len(graph) == 0 or not nx.is_connected(graph):
        return 0.0
    try:
        return float(nx.average_shortest_path_length(graph))
    except nx.NetworkXError:
        return 0.0


def _safe_algebraic_connectivity(graph: nx.Graph) -> float:
    """Return the algebraic connectivity (Fiedler value), or 0.0 on failure."""
    if len(graph) < 2 or not nx.is_connected(graph):
        return 0.0
    try:
        return float(nx.algebraic_connectivity(graph))
    except nx.NetworkXError:
        return 0.0


def _safe_diameter(graph: nx.Graph) -> float:
    """Return the graph diameter, or 0.0 for empty/disconnected graphs."""
    if len(graph) == 0 or not nx.is_connected(graph):
        return 0.0
    try:
        return float(nx.diameter(graph))
    except nx.NetworkXError:
        return 0.0


TARGET_FUNCTIONS: dict[str, Callable[[nx.Graph], float]] = {
    "average_shortest_path_length": _safe_average_shortest_path_length,
    "algebraic_connectivity": _safe_algebraic_connectivity,
    "diameter": _safe_diameter,
}


def target_values(graphs: list[nx.Graph], target_name: str) -> list[float]:
    """Compute target values for a list of graphs.

    Args:
        graphs: List of networkx graphs to evaluate.
        target_name: Key into TARGET_FUNCTIONS (e.g. 'average_shortest_path_length').

    Returns:
        List of float target values, one per graph.

    Raises:
        ValueError: If target_name is not in TARGET_FUNCTIONS.
    """
    target_fn = TARGET_FUNCTIONS.get(target_name)
    if target_fn is None:
        raise ValueError(f"unsupported target: {target_name}")
    return [target_fn(graph) for graph in graphs]
