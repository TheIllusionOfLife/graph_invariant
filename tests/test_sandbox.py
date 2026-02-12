from graph_invariant.sandbox import (
    SandboxEvaluator,
    evaluate_candidate_on_features,
    validate_code_static,
)

_FEATURES_PATH5 = {
    "n": 5,
    "m": 4,
    "density": 0.4,
    "avg_degree": 1.6,
    "max_degree": 2,
    "min_degree": 1,
    "std_degree": 0.48989794855663565,
    "avg_clustering": 0.0,
    "transitivity": 0.0,
    "degree_assortativity": -0.25,
    "num_triangles": 0,
    "degrees": [1, 1, 2, 2, 2],
}


def test_validate_code_static_rejects_forbidden_symbols():
    ok, reason = validate_code_static("import os\ndef new_invariant(s):\n    return 1")
    assert not ok
    assert reason is not None


def test_validate_code_static_allows_simple_function():
    ok, reason = validate_code_static("def new_invariant(s):\n    return s['n']")
    assert ok
    assert reason is None


def test_validate_code_static_rejects_getattr_bypass():
    code = "def new_invariant(s):\n    return getattr(s, 'n')"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_validate_code_static_rejects_non_whitelisted_call():
    code = "def new_invariant(s):\n    return compile('1', '', 'eval')"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_validate_code_static_rejects_forbidden_module_calls():
    code = "def new_invariant(s):\n    return shutil.rmtree('/')"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


# --- nx is now forbidden in sandbox ---


def test_validate_code_static_rejects_nx_attr():
    code = "def new_invariant(s):\n    return nx.density(s)"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_validate_code_static_allows_np_calls():
    code = "def new_invariant(s):\n    return np.mean(s['degrees'])"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_evaluate_with_np_function_runtime():
    code = "def new_invariant(s):\n    return np.mean(s['degrees'])"
    result = evaluate_candidate_on_features(
        code, [_FEATURES_PATH5], timeout_sec=1.0, memory_mb=128
    )
    assert result[0] is not None
    assert abs(result[0] - 1.6) < 0.01


# --- For/While loops and additional builtins ---


def test_validate_code_static_allows_for_loop():
    code = (
        "def new_invariant(s):\n"
        "    total = 0\n"
        "    for d in s['degrees']:\n"
        "        total = total + d\n"
        "    return total"
    )
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_while_loop():
    code = "def new_invariant(s):\n    n = 1\n    while n < 10:\n        n = n + 1\n    return n"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_round_and_list_builtins():
    code = "def new_invariant(s):\n    return round(len(list(s['degrees'])))"
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_allows_local_variable_method_calls():
    code = (
        "def new_invariant(s):\n"
        "    degrees = list(s['degrees'])\n"
        "    degrees.sort()\n"
        "    return float(len(degrees))"
    )
    ok, reason = validate_code_static(code)
    assert ok
    assert reason is None


def test_validate_code_static_still_rejects_forbidden_modules():
    code = "def new_invariant(s):\n    return os.getcwd()"
    ok, reason = validate_code_static(code)
    assert not ok
    assert reason is not None


def test_evaluate_for_loop_computes_correctly():
    code = (
        "def new_invariant(s):\n"
        "    total = 0\n"
        "    for d in s['degrees']:\n"
        "        total = total + d\n"
        "    return float(total)"
    )
    result = evaluate_candidate_on_features(
        code, [_FEATURES_PATH5], timeout_sec=1.0, memory_mb=128
    )
    # path_graph(5): degrees [1,1,2,2,2], sum=8
    assert result[0] is not None
    assert abs(result[0] - 8.0) < 0.01


def test_evaluate_degrees_list_operations():
    code = (
        "def new_invariant(s):\n"
        "    return sum(d ** 2 for d in s['degrees']) / s['n']"
    )
    result = evaluate_candidate_on_features(
        code, [_FEATURES_PATH5], timeout_sec=1.0, memory_mb=128
    )
    # degrees=[1,1,2,2,2], sum of squares=1+1+4+4+4=14, /5=2.8
    assert result[0] is not None
    assert abs(result[0] - 2.8) < 0.01


def test_evaluate_candidate_on_features_times_out_and_returns_none():
    code = "def new_invariant(s):\n    while True:\n        pass"
    result = evaluate_candidate_on_features(
        code, [_FEATURES_PATH5], timeout_sec=0.1, memory_mb=128
    )
    assert result == [None]


def test_evaluate_candidate_on_features_allows_float_and_int_builtins():
    code = "def new_invariant(s):\n    return float(int(s['n']))"
    result = evaluate_candidate_on_features(
        code, [_FEATURES_PATH5], timeout_sec=0.2, memory_mb=128
    )
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
    code = "def new_invariant(s):\n    return 1 / 0"
    with SandboxEvaluator(timeout_sec=0.1, memory_mb=128) as evaluator:
        details = evaluator.evaluate_detailed(code, [_FEATURES_PATH5])
    assert details[0]["value"] is None
    assert details[0]["error_type"] == "runtime_exception"


def test_sandbox_evaluator_reuses_pool_within_context(monkeypatch):
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
        assert evaluator.evaluate(
            "def new_invariant(s):\n    return 1.0", [_FEATURES_PATH5]
        ) == [1.0]
        assert evaluator.evaluate(
            "def new_invariant(s):\n    return 1.0", [{"n": 4}]
        ) == [1.0]

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
        result = evaluator.evaluate(
            "def new_invariant(s):\n    return 1.0", [_FEATURES_PATH5]
        )

    assert result == [2.0]
    assert created_pools == 2
    assert starmap_attempts == 2


def test_compiled_candidate_code_uses_cache():
    from graph_invariant import sandbox

    sandbox._COMPILED_CODE_CACHE = {}
    code = "def new_invariant(s):\n    return 1.0"
    first = sandbox._compiled_candidate_code(code)
    second = sandbox._compiled_candidate_code(code)
    assert first is second
