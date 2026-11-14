"""The worked numbers in the implementation plan."""
import math

import numpy as np
import pytest

from earnedpass.eval import conformal
from earnedpass.eval.bounds import cp_lower, units_needed


@pytest.mark.parametrize(
    "k,n,expected",
    [(1200, 1200, 0.9975), (24, 24, 0.883), (50, 50, 0.942), (1, 1, 0.05), (3, 3, 0.368)],
)
def test_all_success_bounds(k, n, expected):
    assert cp_lower(k, n) == pytest.approx(expected, abs=5e-4)


def test_bound_with_failures_matches_beta():
    assert cp_lower(0, 10) == 0.0
    assert cp_lower(9, 10) == pytest.approx(0.6058, abs=1e-3)
    assert cp_lower(9, 10) < cp_lower(10, 10)


@pytest.mark.parametrize("target,lots", [(0.90, 29), (0.95, 59), (0.99, 299)])
def test_units_needed(target, lots):
    assert units_needed(target) == lots
    assert cp_lower(lots, lots) >= target > cp_lower(lots - 1, lots - 1)


def test_min_alpha():
    assert conformal.min_alpha(30) == pytest.approx(0.032, abs=5e-4)
    assert conformal.min_alpha(3) == 0.25


def test_threshold_position():
    s = np.arange(1, 31) / 100  # 30 calibration units
    assert conformal.threshold(s, 0.10) == 0.28  # ceil(31 * 0.9) = 28
    assert conformal.threshold(s, 0.02) == math.inf  # below 1 / (n + 1): keep every band
    assert conformal.threshold(s[:3], 0.10) == math.inf


def test_sets_keep_bands_at_or_below_q():
    p = np.array([[0.7, 0.2, 0.08, 0.02]])
    assert conformal.sets(p, 0.5).tolist() == [[True, False, False, False]]
    assert conformal.sets(p, 0.8).tolist() == [[True, True, False, False]]
    assert conformal.sets(p, math.inf).all()
