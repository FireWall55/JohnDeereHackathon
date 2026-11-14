"""Split-conformal band sets."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 minus the probability given to the true band."""
    return 1.0 - proba[np.arange(len(y)), y]


def threshold(cal_scores, alpha: float) -> float:
    """The score at position ceil((n + 1)(1 - alpha)); infinite if that exceeds n."""
    s = np.sort(np.asarray(cal_scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-12)
    return math.inf if k > n else float(s[k - 1])


def cell_scores(sample_scores, cells, pct=90) -> pd.Series:
    """One score per cell: the cell's 90th-percentile sample score."""
    return pd.Series(np.asarray(sample_scores)).groupby(np.asarray(cells)).quantile(pct / 100)


def sets(proba: np.ndarray, q: float) -> np.ndarray:
    """Boolean (n, bands): keep every band whose score is at or below q."""
    return (1.0 - proba) <= q + 1e-12


def min_alpha(n_units: int) -> float:
    """With n independent calibration units alpha cannot go below 1 / (n + 1)."""
    return 1.0 / (n_units + 1)
