import math

from graph_invariant.scoring import (
    compute_metrics,
    compute_novelty_bonus,
    compute_simplicity_score,
    compute_total_score,
)


def test_compute_metrics_perfect_monotonic_signal():
    metrics = compute_metrics([1, 2, 3], [10, 20, 30])
    assert math.isclose(metrics.rho_spearman, 1.0, rel_tol=1e-9)
    assert math.isclose(metrics.r_pearson, 1.0, rel_tol=1e-9)
    assert metrics.valid_count == 3
    assert metrics.error_count == 0


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


def test_compute_novelty_bonus_and_total_score():
    bonus = compute_novelty_bonus([1, 2, 3], {"known": [1, 2, 4]})
    assert 0.0 <= bonus <= 1.0
    total = compute_total_score(abs_spearman=0.9, simplicity=0.5, novelty_bonus=0.25)
    assert math.isclose(total, 0.69, rel_tol=1e-9)
