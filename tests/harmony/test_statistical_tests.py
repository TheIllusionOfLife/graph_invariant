"""Tests for analysis.statistical_tests — paired bootstrap CI, Cliff's delta, permutation test.

TDD: written BEFORE implementation. Verifies:
  - paired_bootstrap_ci returns expected structure with valid CIs
  - cliffs_delta returns value in [-1, 1] with correct antisymmetry
  - permutation_test returns valid p-value
  - Same distributions produce non-significant results
  - Clearly separated distributions produce significant results
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))


class TestPairedBootstrapCI:
    def test_returns_expected_keys(self):
        from statistical_tests import paired_bootstrap_ci

        a = [0.5, 0.6, 0.7, 0.55, 0.65, 0.72, 0.58, 0.61, 0.69, 0.63]
        b = [0.4, 0.5, 0.6, 0.45, 0.55, 0.62, 0.48, 0.51, 0.59, 0.53]
        result = paired_bootstrap_ci(a, b, n_bootstrap=500, seed=42)
        assert "mean_diff" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert "p_value" in result

    def test_ci_contains_mean_diff(self):
        from statistical_tests import paired_bootstrap_ci

        a = [0.5, 0.6, 0.7, 0.55, 0.65, 0.72, 0.58, 0.61, 0.69, 0.63]
        b = [0.4, 0.5, 0.6, 0.45, 0.55, 0.62, 0.48, 0.51, 0.59, 0.53]
        result = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=42)
        assert result["ci_low"] <= result["mean_diff"] <= result["ci_high"]

    def test_same_distribution_large_p(self):
        from statistical_tests import paired_bootstrap_ci

        a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        result = paired_bootstrap_ci(a, b, n_bootstrap=500, seed=42)
        assert result["p_value"] >= 0.5  # Very far from significant

    def test_clear_separation_small_p(self):
        from statistical_tests import paired_bootstrap_ci

        a = [0.9, 0.91, 0.92, 0.89, 0.93, 0.88, 0.90, 0.91, 0.92, 0.90]
        b = [0.1, 0.11, 0.12, 0.09, 0.13, 0.08, 0.10, 0.11, 0.12, 0.10]
        result = paired_bootstrap_ci(a, b, n_bootstrap=1000, seed=42)
        assert result["p_value"] < 0.05

    def test_p_value_in_unit_interval(self):
        from statistical_tests import paired_bootstrap_ci

        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        result = paired_bootstrap_ci(a, b, n_bootstrap=200, seed=42)
        assert 0.0 <= result["p_value"] <= 1.0


class TestCliffsDelta:
    def test_bounds(self):
        from statistical_tests import cliffs_delta

        a = [0.9, 0.8, 0.7]
        b = [0.1, 0.2, 0.3]
        d = cliffs_delta(a, b)
        assert -1.0 <= d <= 1.0

    def test_antisymmetry(self):
        from statistical_tests import cliffs_delta

        a = [0.9, 0.8, 0.7, 0.85]
        b = [0.1, 0.2, 0.3, 0.15]
        d_ab = cliffs_delta(a, b)
        d_ba = cliffs_delta(b, a)
        assert d_ab == pytest.approx(-d_ba)

    def test_identical_is_zero(self):
        from statistical_tests import cliffs_delta

        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        assert cliffs_delta(a, b) == pytest.approx(0.0)

    def test_complete_dominance(self):
        from statistical_tests import cliffs_delta

        a = [10.0, 11.0, 12.0]
        b = [1.0, 2.0, 3.0]
        assert cliffs_delta(a, b) == pytest.approx(1.0)


class TestPermutationTest:
    def test_returns_float_in_unit_interval(self):
        from statistical_tests import permutation_test

        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        p = permutation_test(a, b, n_permutations=500, seed=42)
        assert 0.0 <= p <= 1.0

    def test_identical_gives_large_p(self):
        from statistical_tests import permutation_test

        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        p = permutation_test(a, b, n_permutations=500, seed=42)
        assert p >= 0.5

    def test_clear_separation_gives_small_p(self):
        from statistical_tests import permutation_test

        a = [0.9, 0.91, 0.92, 0.89, 0.93, 0.88, 0.90, 0.91, 0.92, 0.90]
        b = [0.1, 0.11, 0.12, 0.09, 0.13, 0.08, 0.10, 0.11, 0.12, 0.10]
        p = permutation_test(a, b, n_permutations=1000, seed=42)
        assert p < 0.05
