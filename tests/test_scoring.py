import math

from graph_invariant.scoring import (
    compute_metrics,
    compute_novelty_bonus,
    compute_novelty_ci,
    compute_simplicity_score,
    compute_total_score,
)


def test_compute_metrics_perfect_monotonic_signal():
    metrics = compute_metrics([1, 2, 3], [10, 20, 30])
    assert math.isclose(metrics.rho_spearman, 1.0, rel_tol=1e-9)
    assert math.isclose(metrics.r_pearson, 1.0, rel_tol=1e-9)
    assert metrics.valid_count == 3
    assert metrics.error_count == 0


def test_compute_metrics_single_sample_is_safe():
    metrics = compute_metrics([1.0], [2.0])
    assert metrics.rho_spearman == 0.0
    assert metrics.r_pearson == 0.0
    assert metrics.valid_count == 1
    assert math.isclose(metrics.rmse, 1.0, rel_tol=1e-9)
    assert math.isclose(metrics.mae, 1.0, rel_tol=1e-9)


def test_compute_metrics_empty_input_is_safe():
    metrics = compute_metrics([], [])
    assert metrics.rho_spearman == 0.0
    assert metrics.r_pearson == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.valid_count == 0


def test_compute_simplicity_score_prefers_shorter_code():
    short_code = "def new_invariant(G):\n    return G.number_of_nodes()"
    long_code = (
        "def new_invariant(G):\n"
        "    n = G.number_of_nodes()\n"
        "    m = G.number_of_edges()\n"
        "    x = (n + m) / (n + 1)\n"
        "    return (x + n + m) / (x + 1)"
    )
    assert compute_simplicity_score(short_code) > compute_simplicity_score(long_code)


def test_compute_simplicity_score_skips_sympy_for_unsafe_expression(monkeypatch):
    calls = {"count": 0}

    def _tracking_simplify(_expr):
        calls["count"] += 1
        return 1

    monkeypatch.setattr("graph_invariant.scoring.sympy.simplify", _tracking_simplify)
    code = "def new_invariant(G):\n    return __import__('os').system('echo unsafe')"
    compute_simplicity_score(code)
    assert calls["count"] == 0


def test_compute_novelty_bonus_and_total_score():
    bonus = compute_novelty_bonus([1, 2, 3], {"known": [1, 2, 4]})
    assert 0.0 <= bonus <= 1.0
    total = compute_total_score(abs_spearman=0.9, simplicity=0.5, novelty_bonus=0.25)
    assert math.isclose(total, 0.69, rel_tol=1e-9)


def test_compute_novelty_ci_reports_bootstrap_upper_bounds():
    result = compute_novelty_ci(
        candidate_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        known_invariants={
            "known_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "known_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        },
        n_bootstrap=128,
        seed=123,
        novelty_threshold=0.7,
    )
    assert "max_ci_upper_abs_rho" in result
    assert "per_invariant" in result
    assert result["max_ci_upper_abs_rho"] >= 0.0
    assert set(result["per_invariant"].keys()) == {"known_a", "known_b"}
    for payload in result["per_invariant"].values():
        assert payload["ci_upper_abs_rho"] >= payload["point_abs_rho"]
