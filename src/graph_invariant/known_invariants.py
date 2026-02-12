import math
from typing import Any

import networkx as nx
import numpy as np


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def compute_feature_dict(graph: nx.Graph) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = sorted(d for _, d in graph.degree())
    avg_degree = (2.0 * m / n) if n > 0 else 0.0
    max_deg = max(degrees) if degrees else 0
    min_deg = min(degrees) if degrees else 0
    std_degree = float(np.std(degrees)) if degrees else 0.0
    try:
        assortativity = nx.degree_assortativity_coefficient(graph)
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        assortativity = 0.0
    triangle_counts = nx.triangles(graph)
    return {
        "n": n,
        "m": m,
        "density": _safe_float(nx.density(graph)),
        "avg_degree": _safe_float(avg_degree),
        "max_degree": max_deg,
        "min_degree": min_deg,
        "std_degree": _safe_float(std_degree),
        "avg_clustering": _safe_float(nx.average_clustering(graph)),
        "transitivity": _safe_float(nx.transitivity(graph)),
        "degree_assortativity": _safe_float(assortativity),
        "num_triangles": sum(triangle_counts.values()) // 3,
        "degrees": degrees,
    }


def compute_feature_dicts(graphs: list[nx.Graph]) -> list[dict[str, Any]]:
    return [compute_feature_dict(g) for g in graphs]


def compute_known_invariant_values(graphs: list[nx.Graph]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {
        "density": [],
        "clustering_coefficient": [],
        "degree_assortativity": [],
        "transitivity": [],
        "average_degree": [],
        "max_degree": [],
        "spectral_radius": [],
        "diameter": [],
        "algebraic_connectivity": [],
    }
    for graph in graphs:
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        degrees = [d for _, d in graph.degree()]
        avg_degree = (2.0 * m / n) if n > 0 else 0.0
        max_degree = float(max(degrees)) if degrees else 0.0
        adjacency = nx.to_numpy_array(graph, dtype=float)
        eigvals = np.linalg.eigvals(adjacency)
        spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        try:
            diameter = float(nx.diameter(graph))
        except nx.NetworkXError:
            diameter = 0.0
        try:
            algebraic = float(nx.algebraic_connectivity(graph))
        except nx.NetworkXError:
            algebraic = 0.0

        out["density"].append(_safe_float(nx.density(graph)))
        out["clustering_coefficient"].append(_safe_float(nx.average_clustering(graph)))
        out["degree_assortativity"].append(_safe_float(nx.degree_assortativity_coefficient(graph)))
        out["transitivity"].append(_safe_float(nx.transitivity(graph)))
        out["average_degree"].append(_safe_float(avg_degree))
        out["max_degree"].append(_safe_float(max_degree))
        out["spectral_radius"].append(_safe_float(spectral_radius))
        out["diameter"].append(_safe_float(diameter))
        out["algebraic_connectivity"].append(_safe_float(algebraic))
    return out
