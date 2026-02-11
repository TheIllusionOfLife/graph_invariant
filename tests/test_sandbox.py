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
    code = "def new_invariant(G):\n    return round(G.number_of_nodes())"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_validate_code_static_rejects_networkx_module_calls():
    code = "def new_invariant(G):\n    return nx.number_of_nodes(G)"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_evaluate_candidate_on_graphs_times_out_and_returns_none():
    import networkx as nx

    code = "def new_invariant(G):\n    while True:\n        pass"
    result = evaluate_candidate_on_graphs(code, [nx.path_graph(5)], timeout_sec=0.1, memory_mb=128)
    assert result == [None]


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
