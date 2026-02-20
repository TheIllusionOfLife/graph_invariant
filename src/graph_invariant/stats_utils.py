from __future__ import annotations

import math
from typing import Any

# 95% two-sided critical t values by degrees of freedom.
# Source: standard t table, rounded to 3 decimals.
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def safe_float(value: Any) -> float | None:
    """Return a float if value is numeric and non-bool, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def mean_std_ci95(values: list[float]) -> dict[str, float | int | None]:
    """Return sample mean/std and 95% CI half-width.

    Uses t critical values for small sample sizes and 1.96 normal approximation
    for df > 30.
    """
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_half_width": None}

    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_half_width": 0.0}

    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    std = math.sqrt(variance)
    critical = _T_CRIT_95.get(n - 1, 1.96)
    ci95_half_width = critical * std / math.sqrt(n)

    return {"n": n, "mean": mean, "std": std, "ci95_half_width": ci95_half_width}
