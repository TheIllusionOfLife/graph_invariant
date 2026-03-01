"""Tests for scripts.run_multi_seed — multi-seed experiment orchestration."""

from __future__ import annotations

import pandas as pd


def _make_single_seed_df(seed: int) -> pd.DataFrame:
    """Create a mock single-seed metrics DataFrame."""
    # Simulate slightly different results per seed
    base = 0.5 + (seed % 10) * 0.01
    data = {
        "random_hits10": [0.1 + base * 0.01, 0.12],
        "freq_hits10": [0.2 + base * 0.01, 0.22],
        "distmult_hits10": [0.4 + base * 0.02, 0.42],
        "harmony_hits10": [0.6 + base * 0.03, 0.62],
        "mrr_random": [0.05, 0.06],
        "mrr_distmult": [0.3, 0.32],
        "mrr_harmony": [0.5 + base * 0.02, 0.52],
    }
    df = pd.DataFrame(data, index=["linear_algebra", "periodic_table"])
    df.index.name = "domain"
    return df


class TestAggregateSeedResults:
    def test_returns_mean_and_std_columns(self) -> None:
        from scripts.run_multi_seed import aggregate_seed_results

        seed_dfs = {s: _make_single_seed_df(s) for s in [42, 123, 456]}
        result = aggregate_seed_results(seed_dfs)
        assert isinstance(result, pd.DataFrame)
        # Should have mean and std for each metric
        for col in ("harmony_hits10_mean", "harmony_hits10_std"):
            assert col in result.columns

    def test_mean_is_average_of_seeds(self) -> None:
        from scripts.run_multi_seed import aggregate_seed_results

        # Use identical DataFrames so mean == the value
        identical_df = _make_single_seed_df(42)
        seed_dfs = {42: identical_df, 123: identical_df.copy(), 456: identical_df.copy()}
        result = aggregate_seed_results(seed_dfs)
        val = identical_df.loc["linear_algebra", "harmony_hits10"]
        assert abs(result.loc["linear_algebra", "harmony_hits10_mean"] - val) < 1e-10

    def test_std_is_zero_for_identical_seeds(self) -> None:
        from scripts.run_multi_seed import aggregate_seed_results

        identical_df = _make_single_seed_df(42)
        seed_dfs = {42: identical_df, 123: identical_df.copy(), 456: identical_df.copy()}
        result = aggregate_seed_results(seed_dfs)
        assert result.loc["linear_algebra", "harmony_hits10_std"] < 1e-10

    def test_preserves_all_domains(self) -> None:
        from scripts.run_multi_seed import aggregate_seed_results

        seed_dfs = {s: _make_single_seed_df(s) for s in [42, 123]}
        result = aggregate_seed_results(seed_dfs)
        assert "linear_algebra" in result.index
        assert "periodic_table" in result.index

    def test_includes_n_seeds(self) -> None:
        from scripts.run_multi_seed import aggregate_seed_results

        seed_dfs = {s: _make_single_seed_df(s) for s in [42, 123, 456]}
        result = aggregate_seed_results(seed_dfs)
        assert "n_seeds" in result.columns
        assert result.loc["linear_algebra", "n_seeds"] == 3


class TestDefaultSeeds:
    def test_ten_seeds_defined(self) -> None:
        from scripts.run_multi_seed import DEFAULT_SEEDS

        assert len(DEFAULT_SEEDS) == 10
        assert all(isinstance(s, int) for s in DEFAULT_SEEDS)

    def test_seeds_are_unique(self) -> None:
        from scripts.run_multi_seed import DEFAULT_SEEDS

        assert len(set(DEFAULT_SEEDS)) == len(DEFAULT_SEEDS)
