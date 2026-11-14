"""Certified bounds behind a PASS."""
from __future__ import annotations

import math

from scipy.stats import beta


def cp_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on a success rate, k successes of n."""
    if n <= 0 or k <= 0:
        return 0.0
    if k == n:
        return (1 - conf) ** (1 / n)
    return float(beta.ppf(1 - conf, k, n - k + 1))


def units_needed(target: float, conf: float = 0.95) -> int:
    """Independent units, all successes, needed to certify `target` at `conf`."""
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1")
    return math.ceil(math.log(1 - conf) / math.log(target) - 1e-12)
