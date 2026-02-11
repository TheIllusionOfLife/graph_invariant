from graph_invariant.sandbox import evaluate_candidate_on_graphs, validate_code_static


def test_validate_code_static_rejects_forbidden_symbols():
    ok, reason = validate_code_static("import os\ndef new_invariant(G):\n    return 1")
    assert not ok
    assert reason is not None


def test_validate_code_static_allows_simple_function():
    ok, reason = validate_code_static("def new_invariant(G):\n    return G.number_of_nodes()")
    assert ok
    assert reason is None


def test_validate_code_static_rejects_getattr_bypass():
    code = "def new_invariant(G):\n    return getattr(G, 'number_of_nodes')()"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_evaluate_candidate_on_graphs_times_out_and_returns_none():
    import networkx as nx

    code = "def new_invariant(G):\n    while True:\n        pass"
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=0.1, memory_mb=128)
    assert result == [None]
