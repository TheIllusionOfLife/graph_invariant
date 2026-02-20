from graph_invariant.stats_utils import mean_std_ci95, safe_float


def test_safe_float_filters_bool_and_accepts_numeric() -> None:
    assert safe_float(1) == 1.0
    assert safe_float(1.5) == 1.5
    assert safe_float(True) is None
    assert safe_float("1") is None


def test_mean_std_ci95_uses_t_critical_for_small_n() -> None:
    values = [0.1, 0.2, 0.3]
    stats = mean_std_ci95(values)
    # With n=3, df=2, t_crit=4.303 should be used (not 1.96).
    assert stats["n"] == 3
    assert stats["ci95_half_width"] is not None
    assert float(stats["ci95_half_width"]) > 0.2


def test_mean_std_ci95_handles_empty_and_singleton() -> None:
    assert mean_std_ci95([]) == {
        "n": 0,
        "mean": None,
        "std": None,
        "ci95_half_width": None,
    }
    assert mean_std_ci95([0.5]) == {
        "n": 1,
        "mean": 0.5,
        "std": 0.0,
        "ci95_half_width": 0.0,
    }
