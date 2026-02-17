import networkx as nx
import numpy as np
import pytest

from graph_invariant.known_invariants import (
    compute_feature_dict,
    compute_feature_dicts,
    compute_known_invariant_values,
)

EXPECTED_KEYS = {
    "n",
    "m",
    "density",
    "avg_degree",
    "max_degree",
    "min_degree",
    "std_degree",
    "avg_clustering",
    "transitivity",
    "degree_assortativity",
    "num_triangles",
    "degrees",
    "laplacian_lambda2",
    "laplacian_lambda_max",
    "laplacian_spectral_gap",
    "normalized_laplacian_lambda2",
    "laplacian_energy_ratio",
}


def test_compute_feature_dict_returns_expected_keys():
    g = nx.path_graph(5)
    features = compute_feature_dict(g)
    assert set(features.keys()) == EXPECTED_KEYS


def test_compute_feature_dict_values_for_path_graph():
    g = nx.path_graph(5)
    features = compute_feature_dict(g)
    assert features["n"] == 5
    assert features["m"] == 4
    assert features["degrees"] == [1, 1, 2, 2, 2]
    assert features["max_degree"] == 2
    assert features["min_degree"] == 1
    assert features["num_triangles"] == 0
    assert abs(features["avg_degree"] - 1.6) < 1e-9
    assert abs(features["density"] - 0.4) < 1e-9
    assert features["avg_clustering"] == 0.0
    assert features["transitivity"] == 0.0


def test_compute_feature_dict_single_node():
    g = nx.Graph()
    g.add_node(0)
    features = compute_feature_dict(g)
    assert features["n"] == 1
    assert features["m"] == 0
    assert features["degrees"] == [0]
    assert features["max_degree"] == 0
    assert features["min_degree"] == 0
    assert features["avg_degree"] == 0.0
    assert features["density"] == 0.0
    assert features["std_degree"] == 0.0
    assert features["degree_assortativity"] == 0.0


def test_compute_feature_dicts_batch():
    graphs = [nx.path_graph(3), nx.cycle_graph(4), nx.complete_graph(5)]
    results = compute_feature_dicts(graphs)
    assert len(results) == 3
    assert results[0]["n"] == 3
    assert results[1]["n"] == 4
    assert results[2]["n"] == 5


# ── compute_known_invariant_values tests ─────────────────────────────

EXPECTED_INVARIANT_KEYS = {
    "density",
    "clustering_coefficient",
    "degree_assortativity",
    "transitivity",
    "average_degree",
    "max_degree",
    "spectral_radius",
    "diameter",
    "algebraic_connectivity",
    "laplacian_lambda2",
    "laplacian_lambda_max",
    "laplacian_spectral_gap",
    "normalized_laplacian_lambda2",
    "laplacian_energy_ratio",
}


def test_compute_known_invariant_values_returns_expected_keys():
    graphs = [nx.path_graph(5)]
    result = compute_known_invariant_values(graphs)
    assert set(result.keys()) == EXPECTED_INVARIANT_KEYS


def test_compute_known_invariant_values_list_lengths_match_input():
    graphs = [nx.path_graph(5), nx.cycle_graph(6), nx.complete_graph(4)]
    result = compute_known_invariant_values(graphs)
    for key in EXPECTED_INVARIANT_KEYS:
        assert len(result[key]) == 3, f"length mismatch for {key}"


def test_compute_known_invariant_values_complete_graph():
    """K_5: spectral_radius=4.0, algebraic_connectivity=5.0, diameter=1."""
    result = compute_known_invariant_values([nx.complete_graph(5)])
    assert result["spectral_radius"][0] == pytest.approx(4.0, abs=1e-6)
    assert result["algebraic_connectivity"][0] == pytest.approx(5.0, abs=1e-6)
    assert result["diameter"][0] == 1.0


def test_compute_known_invariant_values_path_graph():
    """P_5: spectral_radius ≈ 1.732, algebraic_connectivity ≈ 0.382."""
    result = compute_known_invariant_values([nx.path_graph(5)])
    # spectral_radius of P_5 = 2*cos(pi/5) ≈ 1.618... wait, let me compute.
    # Eigenvalues of P_n adjacency matrix: 2*cos(k*pi/(n+1)) for k=1..n
    # P_5: 2*cos(pi/6)=sqrt(3)≈1.732, 2*cos(2pi/6)=1, 2*cos(3pi/6)=0,
    #       2*cos(4pi/6)=-1, 2*cos(5pi/6)=-sqrt(3)
    assert result["spectral_radius"][0] == pytest.approx(2 * np.cos(np.pi / 6), abs=1e-4)
    # Algebraic connectivity of P_n = 2*(1 - cos(pi/n))
    expected_alg_conn = 2 * (1 - np.cos(np.pi / 5))
    assert result["algebraic_connectivity"][0] == pytest.approx(expected_alg_conn, abs=1e-4)


def test_compute_known_invariant_values_disconnected_graph():
    """Disconnected graph: diameter=0, algebraic_connectivity=0."""
    g = nx.Graph()
    g.add_nodes_from(range(10))
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    result = compute_known_invariant_values([g])
    assert result["diameter"][0] == 0.0
    assert result["algebraic_connectivity"][0] == 0.0


def test_compute_known_invariant_values_large_graph_completes():
    """A graph with n=300 should complete without hanging."""
    g = nx.barabasi_albert_graph(300, 3, seed=42)
    result = compute_known_invariant_values([g])
    assert result["spectral_radius"][0] > 0
    assert result["algebraic_connectivity"][0] > 0


def test_compute_known_invariant_values_single_node():
    g = nx.Graph()
    g.add_node(0)
    result = compute_known_invariant_values([g])
    assert result["spectral_radius"][0] == 0.0
    assert result["algebraic_connectivity"][0] == 0.0
    assert result["diameter"][0] == 0.0


def test_compute_known_invariant_values_spectral_pack_disabled():
    result = compute_known_invariant_values(
        [nx.path_graph(5)],
        include_spectral_feature_pack=False,
    )
    assert "laplacian_lambda2" not in result
    assert "laplacian_lambda_max" not in result
    assert "laplacian_spectral_gap" not in result
    assert "normalized_laplacian_lambda2" not in result
    assert "laplacian_energy_ratio" not in result
