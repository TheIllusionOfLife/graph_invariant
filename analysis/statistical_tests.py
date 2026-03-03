"""Statistical tests for paired comparison of methods.

Provides:
  - paired_bootstrap_ci: Bootstrap confidence interval for mean difference
  - cliffs_delta: Non-parametric effect size measure
  - permutation_test: Exact (or approximate) permutation test for mean difference
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap confidence interval for mean(a) - mean(b).

    Parameters
    ----------
    scores_a, scores_b:
        Paired score lists (must be same length).
    n_bootstrap:
        Number of bootstrap resamples.
    confidence:
        Confidence level (default 0.95 for 95% CI).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Dict with: mean_diff, ci_low, ci_high, p_value.
    The p-value is the fraction of bootstrap resamples where the
    difference has opposite sign to the observed difference (two-sided).
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"scores_a and scores_b must have equal length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("scores_a and scores_b must be non-empty")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be > 0, got {n_bootstrap}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    n = len(a)
    diffs = a - b
    observed_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[idx])

    alpha_ci = 1.0 - confidence
    ci_low = float(np.percentile(boot_diffs, 100 * alpha_ci / 2))
    ci_high = float(np.percentile(boot_diffs, 100 * (1 - alpha_ci / 2)))

    # Two-sided p-value
    if observed_diff == 0.0:
        p_value = 1.0
    else:
        left = float(np.mean(boot_diffs <= 0))
        right = float(np.mean(boot_diffs >= 0))
        tail = left if observed_diff > 0 else right
        p_value = min(1.0, 2.0 * tail)

    return {
        "mean_diff": observed_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def cliffs_delta(
    scores_a: list[float],
    scores_b: list[float],
) -> float:
    """Cliff's delta: non-parametric effect size in [-1, 1].

    δ = (#{a_i > b_j} - #{a_i < b_j}) / (n_a * n_b)

    +1 = all a > all b (complete dominance of a)
    -1 = all a < all b (complete dominance of b)
     0 = no systematic difference
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    n_a = len(a)
    n_b = len(b)
    if n_a == 0 or n_b == 0:
        return 0.0

    # Vectorised comparison
    comparisons = np.sign(a[:, np.newaxis] - b[np.newaxis, :])
    return float(np.sum(comparisons) / (n_a * n_b))


def permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test for difference in means.

    Returns the p-value: fraction of permuted differences at least as
    extreme as the observed difference.
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("scores_a and scores_b must be non-empty")
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be > 0, got {n_permutations}")
    combined = np.concatenate([a, b])
    n_a = len(a)
    observed_diff = abs(float(np.mean(a) - np.mean(b)))

    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(float(np.mean(combined[:n_a]) - np.mean(combined[n_a:])))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations
