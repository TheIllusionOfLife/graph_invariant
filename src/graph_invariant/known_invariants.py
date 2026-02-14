import math
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _spectral_radius_sparse(graph: nx.Graph) -> float:
    """Compute the spectral radius using sparse eigensolvers.

    For small graphs (n < 10), falls back to dense eigvals since sparse
    solvers require k < n. For larger graphs, uses ARPACK to find the
    largest-magnitude eigenvalue only — O(n·k²) vs O(n³) dense.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return 0.0
    if n < 10:
        adjacency = nx.to_numpy_array(graph, dtype=float)
        eigvals = np.linalg.eigvals(adjacency)
        return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    adj_sparse = nx.to_scipy_sparse_array(graph, dtype=float)
    try:
        vals = scipy.sparse.linalg.eigsh(adj_sparse, k=1, which="LM", maxiter=500)
        return float(np.abs(vals[0][0]))
    except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError):
        adjacency = nx.to_numpy_array(graph, dtype=float)
        eigvals = np.linalg.eigvals(adjacency)
        return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def _algebraic_connectivity_sparse(graph: nx.Graph) -> float:
    """Compute algebraic connectivity (Fiedler value) using sparse Laplacian.

    The Fiedler value is the second-smallest eigenvalue of the graph Laplacian.
    Returns 0.0 for disconnected or trivially small graphs.
    """
    n = graph.number_of_nodes()
    if n < 2 or not nx.is_connected(graph):
        return 0.0
    if n < 10:
        try:
            return float(nx.algebraic_connectivity(graph))
        except nx.NetworkXError:
            return 0.0
    laplacian = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    laplacian = scipy.sparse.csgraph.laplacian(laplacian)
    try:
        vals = scipy.sparse.linalg.eigsh(laplacian, k=2, which="SM", maxiter=1000)
        eigenvalues = sorted(vals[0])
        return max(float(eigenvalues[1]), 0.0)
    except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError):
        try:
            return float(nx.algebraic_connectivity(graph))
        except nx.NetworkXError:
            return 0.0


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
        spectral_radius = _spectral_radius_sparse(graph)
        try:
            diameter = float(nx.diameter(graph))
        except nx.NetworkXError:
            diameter = 0.0
        algebraic = _algebraic_connectivity_sparse(graph)

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
