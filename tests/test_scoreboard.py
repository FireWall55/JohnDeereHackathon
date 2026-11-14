import math

import numpy as np
import pandas as pd

from earnedpass import scoreboard


def table(lanes, levels, top=None, covered=None):
    n = len(lanes)
    return pd.DataFrame({"lane": lanes, "level": levels, "cell": [f"c{l}" for l in levels],
                         "top_band": top or ["zero"] * n, "covered": covered or [True] * n})


def test_headline_counts_each_error_once():
    df = table(["PASS", "HOLD", "LAB", "HOLD", "PASS", "LAB"], [0, 0, 0, 6, 12, 15],
               covered=[True, True, False, True, True, True])
    m = scoreboard.from_lanes(df)
    assert m["wrong_pass"] == 1 / 3  # one of three blends passed
    assert m["pure_held"] == 1 / 3  # one of three pure samples held
    assert m["decided"] == 4 / 6
    assert m["decided_right"] == 2 / 4  # PASS on pure and HOLD on a blend
    assert m["range_has_truth"] == 5 / 6
    assert m["n"] == 6 and m["materials"] == 4


def test_forced_baseline_decides_everything():
    df = table(["LAB"] * 4, [0, 0, 6, 15], top=["zero", "mid", "zero", "high"])
    b = scoreboard.forced(df)
    assert b["decided"] == 1 and b["wrong_pass"] == 0.5 and b["pure_held"] == 0.5
    assert math.isnan(b["range_has_truth"])


def test_amount_and_filler_skip_unscored_rows():
    est = np.array([1.0, np.nan, 10.0])
    assert scoreboard.amount_error(est, [0, 1, 2], np.array([0, 3, 6])) == 2.5
    probs = np.array([[0.9, 0.1, 0], [np.nan] * 3, [0.2, 0.7, 0.1]])
    assert scoreboard.filler_right(probs, [0, 1, 2], np.array([0, 3, 6]), np.array([0, 0, 1])) == 1.0
