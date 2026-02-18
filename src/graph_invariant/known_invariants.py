import math
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg

_SPARSE_EIGEN_FALLBACK_EXCEPTIONS = (
    scipy.sparse.linalg.ArpackNoConvergence,
    scipy.sparse.linalg.ArpackError,
    ValueError,
    RuntimeError,
    TypeError,
)


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def _dense_laplacian_eigs(graph: nx.Graph) -> np.ndarray:
    lap = nx.laplacian_matrix(graph).toarray().astype(float)
    vals = np.linalg.eigvalsh(lap)
    return np.sort(vals)


def _dense_normalized_laplacian_eigs(graph: nx.Graph) -> np.ndarray:
    lap = nx.normalized_laplacian_matrix(graph).toarray().astype(float)
    vals = np.linalg.eigvalsh(lap)
    return np.sort(vals)


def _laplacian_sparse(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    adjacency = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    return scipy.sparse.csgraph.laplacian(adjacency, normed=False).tocsr()


def _normalized_laplacian_sparse(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    return nx.normalized_laplacian_matrix(graph).tocsr().astype(float)


def _laplacian_extrema_sparse(graph: nx.Graph) -> tuple[float, float]:
    """Return (lambda2, lambda_max) of the combinatorial Laplacian."""
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0, 0.0

    if n < 10:
        eigs = _dense_laplacian_eigs(graph)
        return max(float(eigs[1]), 0.0), max(float(eigs[-1]), 0.0)

    lap = _laplacian_sparse(graph)
    try:
        smallest = scipy.sparse.linalg.eigsh(lap, k=2, which="SM", return_eigenvectors=False)
        largest = scipy.sparse.linalg.eigsh(lap, k=1, which="LA", return_eigenvectors=False)
        smallest_sorted = np.sort(smallest)
        lambda2 = max(float(smallest_sorted[1]), 0.0)
        lambda_max = max(float(largest[0]), 0.0)
        return lambda2, lambda_max
    except _SPARSE_EIGEN_FALLBACK_EXCEPTIONS:
        eigs = _dense_laplacian_eigs(graph)
        return max(float(eigs[1]), 0.0), max(float(eigs[-1]), 0.0)


def _normalized_laplacian_lambda2_sparse(graph: nx.Graph) -> float:
    """Return lambda2 of the normalized Laplacian."""
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0

    if n < 10:
        eigs = _dense_normalized_laplacian_eigs(graph)
        return max(float(eigs[1]), 0.0)

    lap = _normalized_laplacian_sparse(graph)
    try:
        smallest = scipy.sparse.linalg.eigsh(lap, k=2, which="SM", return_eigenvectors=False)
        smallest_sorted = np.sort(smallest)
        return max(float(smallest_sorted[1]), 0.0)
    except _SPARSE_EIGEN_FALLBACK_EXCEPTIONS:
        eigs = _dense_normalized_laplacian_eigs(graph)
        return max(float(eigs[1]), 0.0)


def _laplacian_energy_ratio(graph: nx.Graph, k: int = 5) -> float:
    """Return sum(top-k Laplacian eigenvalues) / trace(L), clamped to [0, 1]."""
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0

    trace = 2.0 * graph.number_of_edges()
    if trace <= 0.0:
        return 0.0

    k_eff = max(1, min(k, n))
    if n < 10 or k_eff >= n:
        eigs = _dense_laplacian_eigs(graph)
        top_sum = float(np.sum(eigs[-k_eff:]))
        return max(0.0, min(1.0, top_sum / trace))

    lap = _laplacian_sparse(graph)
    try:
        largest = scipy.sparse.linalg.eigsh(
            lap,
            k=k_eff,
            which="LA",
            return_eigenvectors=False,
        )
        top_sum = float(np.sum(np.maximum(largest, 0.0)))
        return max(0.0, min(1.0, top_sum / trace))
    except _SPARSE_EIGEN_FALLBACK_EXCEPTIONS:
        eigs = _dense_laplacian_eigs(graph)
        top_sum = float(np.sum(eigs[-k_eff:]))
        return max(0.0, min(1.0, top_sum / trace))


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
    except _SPARSE_EIGEN_FALLBACK_EXCEPTIONS:
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
    # Keep this aligned with the non-normalized Laplacian lambda2 key for
    # consistent semantics across exported invariants.
    if n >= 10:
        lambda2, _ = _laplacian_extrema_sparse(graph)
        return max(float(lambda2), 0.0)
    if n < 10:
        try:
            return float(nx.algebraic_connectivity(graph))
        except nx.NetworkXError:
            return 0.0
    return 0.0


def _spectral_pack(graph: nx.Graph) -> dict[str, float]:
    lambda2, lambda_max = _laplacian_extrema_sparse(graph)
    normalized_lambda2 = _normalized_laplacian_lambda2_sparse(graph)
    return {
        "laplacian_lambda2": _safe_float(lambda2),
        "laplacian_lambda_max": _safe_float(lambda_max),
        "laplacian_spectral_gap": _safe_float(max(lambda_max - lambda2, 0.0)),
        "normalized_laplacian_lambda2": _safe_float(normalized_lambda2),
        "laplacian_energy_ratio": _safe_float(_laplacian_energy_ratio(graph, k=5)),
    }


def compute_feature_dict(
    graph: nx.Graph,
    include_spectral_feature_pack: bool = True,
) -> dict[str, Any]:
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
    base = {
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
    if include_spectral_feature_pack:
        base.update(_spectral_pack(graph))
    return base


def compute_feature_dicts(
    graphs: list[nx.Graph],
    include_spectral_feature_pack: bool = True,
) -> list[dict[str, Any]]:
    return [
        compute_feature_dict(g, include_spectral_feature_pack=include_spectral_feature_pack)
        for g in graphs
    ]


def compute_known_invariant_values(
    graphs: list[nx.Graph],
    include_spectral_feature_pack: bool = True,
) -> dict[str, list[float]]:
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
    if include_spectral_feature_pack:
        out.update(
            {
                "laplacian_lambda2": [],
                "laplacian_lambda_max": [],
                "laplacian_spectral_gap": [],
                "normalized_laplacian_lambda2": [],
                "laplacian_energy_ratio": [],
            }
        )

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

        try:
            degree_assortativity = nx.degree_assortativity_coefficient(graph)
        except (nx.NetworkXError, ValueError, ZeroDivisionError):
            degree_assortativity = 0.0
        out["density"].append(_safe_float(nx.density(graph)))
        out["clustering_coefficient"].append(_safe_float(nx.average_clustering(graph)))
        out["degree_assortativity"].append(_safe_float(degree_assortativity))
        out["transitivity"].append(_safe_float(nx.transitivity(graph)))
        out["average_degree"].append(_safe_float(avg_degree))
        out["max_degree"].append(_safe_float(max_degree))
        out["spectral_radius"].append(_safe_float(spectral_radius))
        out["diameter"].append(_safe_float(diameter))
        out["algebraic_connectivity"].append(_safe_float(algebraic))

        if include_spectral_feature_pack:
            pack = _spectral_pack(graph)
            out["laplacian_lambda2"].append(pack["laplacian_lambda2"])
            out["laplacian_lambda_max"].append(pack["laplacian_lambda_max"])
            out["laplacian_spectral_gap"].append(pack["laplacian_spectral_gap"])
            out["normalized_laplacian_lambda2"].append(pack["normalized_laplacian_lambda2"])
            out["laplacian_energy_ratio"].append(pack["laplacian_energy_ratio"])

    return out
