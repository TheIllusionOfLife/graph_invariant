import math

import pytest

from graph_invariant.scoring import (
    compute_bound_metrics,
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
    total = compute_total_score(fitness_signal=0.9, simplicity=0.5, novelty_bonus=0.25)
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


# ── BoundMetrics tests ───────────────────────────────────────────────


def test_compute_bound_metrics_perfect_upper_bound():
    """f(x) >= y for all points → satisfaction_rate=1.0, tight bound scores well."""
    # y_true = [1, 2, 3], y_pred = [1.5, 2.5, 3.5] → all satisfy f(x) >= y
    result = compute_bound_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], mode="upper_bound")
    assert result.satisfaction_rate == 1.0
    assert result.violation_count == 0
    assert result.valid_count == 3
    assert result.bound_score > 0.0
    assert result.mean_gap == pytest.approx(0.5, abs=1e-9)


def test_compute_bound_metrics_perfect_lower_bound():
    """f(x) <= y for all points → satisfaction_rate=1.0."""
    result = compute_bound_metrics([3.0, 4.0, 5.0], [2.0, 3.0, 4.0], mode="lower_bound")
    assert result.satisfaction_rate == 1.0
    assert result.violation_count == 0
    assert result.valid_count == 3
    assert result.bound_score > 0.0


def test_compute_bound_metrics_violating_upper_bound():
    """One violation tanks the satisfaction rate."""
    # y_true = [1, 2, 3], y_pred = [0.5, 2.5, 3.5] → first point violates f(x) >= y
    result = compute_bound_metrics([1.0, 2.0, 3.0], [0.5, 2.5, 3.5], mode="upper_bound")
    assert result.satisfaction_rate == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert result.violation_count == 1


def test_compute_bound_metrics_all_violating():
    """All points violate → score is 0."""
    result = compute_bound_metrics([5.0, 6.0, 7.0], [1.0, 2.0, 3.0], mode="upper_bound")
    assert result.satisfaction_rate == 0.0
    assert result.bound_score == 0.0
    assert result.violation_count == 3


def test_compute_bound_metrics_tighter_bound_scores_higher():
    """Among two fully-satisfying upper bounds, the tighter one scores higher."""
    tight = compute_bound_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], mode="upper_bound")
    loose = compute_bound_metrics([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], mode="upper_bound")
    assert tight.bound_score > loose.bound_score


def test_compute_bound_metrics_empty_input():
    """Empty input should return safe defaults."""
    result = compute_bound_metrics([], [], mode="upper_bound")
    assert result.satisfaction_rate == 0.0
    assert result.bound_score == 0.0
    assert result.valid_count == 0


def test_compute_bound_metrics_respects_tolerance():
    """A tiny violation within tolerance should still count as satisfied."""
    # f(x) = 0.9999999999 vs y = 1.0 — within default epsilon
    result = compute_bound_metrics([1.0], [1.0 - 1e-10], mode="upper_bound", tolerance=1e-9)
    assert result.satisfaction_rate == 1.0
    assert result.violation_count == 0
