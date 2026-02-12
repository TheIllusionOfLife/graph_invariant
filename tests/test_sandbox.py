from graph_invariant.sandbox import (
    SandboxEvaluator,
    evaluate_candidate_on_graphs,
    validate_code_static,
)


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


def test_validate_code_static_rejects_non_whitelisted_call():
    code = "def new_invariant(G):\n    return compile('1', '', 'eval')"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_validate_code_static_rejects_forbidden_module_calls():
    code = "def new_invariant(G):\n    return shutil.rmtree('/')"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


# --- P0: nx and np should be allowed in sandbox ---


def test_validate_code_static_allows_nx_module_calls():
    code = "def new_invariant(G):\n    return nx.density(G)"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_np_calls():
    code = "def new_invariant(G):\n    return np.mean([1, 2, 3])"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_evaluate_with_nx_function_runtime():
    import networkx as nx

    code = "def new_invariant(G):\n    return nx.density(G)"
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=1.0, memory_mb=128)
    assert result[0] is not None
    assert abs(result[0] - 0.4) < 0.01


def test_evaluate_with_np_function_runtime():
    import networkx as nx

    code = (
        "def new_invariant(G):\n"
        "    degrees = [d for _, d in G.degree()]\n"
        "    return np.mean(degrees)"
    )
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=1.0, memory_mb=128)
    assert result[0] is not None
    assert abs(result[0] - 1.6) < 0.01


# --- P1: For/While loops and additional builtins ---


def test_validate_code_static_allows_for_loop():
    code = (
        "def new_invariant(G):\n"
        "    s = 0\n"
        "    for _, d in G.degree():\n"
        "        s = s + d\n"
        "    return s"
    )
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_while_loop():
    code = "def new_invariant(G):\n    n = 1\n    while n < 10:\n        n = n + 1\n    return n"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_round_and_list_builtins():
    code = "def new_invariant(G):\n    return round(len(list(G.degree())))"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_local_variable_method_calls():
    code = (
        "def new_invariant(G):\n"
        "    degrees = list(G.degree())\n"
        "    degrees.sort()\n"
        "    return float(len(degrees))"
    )
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_still_rejects_forbidden_modules():
    code = "def new_invariant(G):\n    return os.getcwd()"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_validate_code_static_rejects_target_function():
    code = "def new_invariant(G):\n    return nx.average_shortest_path_length(G)"
    ok, reason = validate_code_static(code, target_name="average_shortest_path_length")
    assert not ok
    assert "target" in reason.lower()


def test_validate_code_static_rejects_related_shortcut_functions():
    code = "def new_invariant(G):\n    return nx.shortest_path_length(G, 0, 1)"
    ok, reason = validate_code_static(code, target_name="average_shortest_path_length")
    assert not ok
    assert reason is not None


def test_validate_code_static_allows_unrelated_nx_calls_with_target():
    code = "def new_invariant(G):\n    return nx.density(G)"
    ok, reason = validate_code_static(code, target_name="average_shortest_path_length")
    assert ok
    assert reason is None


def test_evaluate_for_loop_computes_correctly():
    import networkx as nx

    code = (
        "def new_invariant(G):\n"
        "    s = 0\n"
        "    for _, d in G.degree():\n"
        "        s = s + d\n"
        "    return float(s)"
    )
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=1.0, memory_mb=128)
    # path_graph(5): degrees [1,2,2,2,1], sum=8
    assert result[0] is not None
    assert abs(result[0] - 8.0) < 0.01


def test_evaluate_candidate_on_graphs_times_out_and_returns_none():
    import networkx as nx

    code = "def new_invariant(G):\n    while True:\n        pass"
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=0.1, memory_mb=128)
    assert result == [None]


def test_evaluate_candidate_on_graphs_allows_float_and_int_builtins():
    import networkx as nx

    code = "def new_invariant(G):\n    return float(int(G.number_of_nodes()))"
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=0.2, memory_mb=128)
    assert result == [5.0]


def test_sandbox_evaluate_detailed_reports_static_invalid_without_pool():
    evaluator = SandboxEvaluator(timeout_sec=0.1, memory_mb=128)
    details = evaluator.evaluate_detailed("import os\n", [])
    assert details == []

    details = evaluator.evaluate_detailed("import os\n", [object()])
    assert details[0]["value"] is None
    assert details[0]["error_type"] == "static_invalid"
    assert "forbidden token" in str(details[0]["error_detail"])


def test_sandbox_evaluate_detailed_reports_runtime_exception():
    import networkx as nx

    code = "def new_invariant(G):\n    return 1 / 0"
    with SandboxEvaluator(timeout_sec=0.1, memory_mb=128) as evaluator:
        details = evaluator.evaluate_detailed(code, [nx.path_graph(5)])
    assert details[0]["value"] is None
    assert details[0]["error_type"] == "runtime_exception"


def test_sandbox_evaluator_reuses_pool_within_context(monkeypatch):
    import networkx as nx

    created_pools = 0
    starmap_calls = 0

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def starmap(self, _fn, tasks, chunksize=1):
            del chunksize
            nonlocal starmap_calls
            starmap_calls += 1
            return [1.0 for _ in tasks]

        def close(self):
            return None

        def join(self):
            return None

    class FakeContext:
        def Pool(self, **kwargs):  # noqa: N802
            del kwargs
            nonlocal created_pools
            created_pools += 1
            return FakePool()

    monkeypatch.setattr("graph_invariant.sandbox.mp.get_context", lambda: FakeContext())

    with SandboxEvaluator(timeout_sec=0.1, memory_mb=128) as evaluator:
        assert evaluator.evaluate("def new_invariant(G):\n    return 1.0", [nx.path_graph(4)]) == [
            1.0
        ]
        assert evaluator.evaluate("def new_invariant(G):\n    return 1.0", [nx.path_graph(5)]) == [
            1.0
        ]

    assert created_pools == 1
    assert starmap_calls == 2


def test_sandbox_evaluator_does_not_create_pool_until_first_evaluate(monkeypatch):
    class FakeContext:
        def Pool(self, **kwargs):  # noqa: N802
            del kwargs
            raise AssertionError("Pool should not be created")

    monkeypatch.setattr("graph_invariant.sandbox.mp.get_context", lambda: FakeContext())

    with SandboxEvaluator(timeout_sec=0.1, memory_mb=128):
        pass


def test_sandbox_evaluator_rebuilds_pool_after_broken_pipe(monkeypatch):
    import networkx as nx

    created_pools = 0
    starmap_attempts = 0

    class BrokenPool:
        def starmap(self, _fn, _tasks, chunksize=1):
            del _fn, _tasks, chunksize
            nonlocal starmap_attempts
            starmap_attempts += 1
            raise BrokenPipeError

        def close(self):
            return None

        def join(self):
            return None

    class HealthyPool:
        def starmap(self, _fn, tasks, chunksize=1):
            del _fn, chunksize
            nonlocal starmap_attempts
            starmap_attempts += 1
            return [2.0 for _ in tasks]

        def close(self):
            return None

        def join(self):
            return None

    class FakeContext:
        def Pool(self, **kwargs):  # noqa: N802
            del kwargs
            nonlocal created_pools
            created_pools += 1
            if created_pools == 1:
                return BrokenPool()
            return HealthyPool()

    monkeypatch.setattr("graph_invariant.sandbox.mp.get_context", lambda: FakeContext())

    with SandboxEvaluator(timeout_sec=0.1, memory_mb=128) as evaluator:
        result = evaluator.evaluate("def new_invariant(G):\n    return 1.0", [nx.path_graph(4)])

    assert result == [2.0]
    assert created_pools == 2
    assert starmap_attempts == 2


def test_compiled_candidate_code_uses_cache():
    from graph_invariant import sandbox

    sandbox._COMPILED_CODE_CACHE = {}
    code = "def new_invariant(G):\n    return 1.0"
    first = sandbox._compiled_candidate_code(code)
    second = sandbox._compiled_candidate_code(code)
    assert first is second
