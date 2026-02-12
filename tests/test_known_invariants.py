import networkx as nx

from graph_invariant.known_invariants import compute_feature_dict, compute_feature_dicts

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
