import numpy as np
import pandas as pd

from earnedpass import verdict as V


def present(peanut, walnut, wood):
    return np.array([[peanut, walnut, wood]], dtype=float)  # folder order P, W, D


def lane(peanut, walnut, wood, th=V.Thresholds()):
    l, lp, lw = V.rules(present(peanut, walnut, wood), th)
    return l[0], bool(lp[0]), bool(lw[0])


def test_team_rules():
    assert lane(0.0, 0.0, 0.50) == (V.REJECT, False, False)
    assert lane(0.0, 0.0, 0.10)[0] == V.REJECT
    assert lane(0.0, 0.0, 0.02)[0] == V.LAB_WOOD  # 2% is not above the reject limit
    assert lane(0.0, 0.0, 0.015)[0] == V.LAB_WOOD
    assert lane(0.0, 0.0, 0.01)[0] == V.LAB_PURE  # 1% is not above the lab limit
    assert lane(0.40, 0.0, 0.0) == (V.LABEL, True, False)
    assert lane(0.0, 0.40, 0.0) == (V.LABEL, False, True)
    assert lane(0.30, 0.30, 0.0) == (V.LABEL, True, True)  # both labelled, no choice made
    assert lane(0.05, 0.05, 0.005) == (V.LAB_PURE, False, False)
    assert lane(0.50, 0.0, 0.20) == (V.REJECT, True, False)  # wood decides; the nut label is still reported


def test_thresholds_are_adjustable():
    th = V.Thresholds(wood_lab=0.05, wood_reject=0.30, peanut=0.50, walnut=0.50)
    assert lane(0.40, 0.0, 0.04, th)[0] == V.LAB_PURE
    assert lane(0.0, 0.0, 0.20, th)[0] == V.LAB_WOOD


def test_chance_present_needs_filler_to_be_likely():
    p_band = np.array([[0.95, 0.05, 0, 0], [0.0, 0.5, 0.5, 0]])
    p_type = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    p_any, pres = V.chances(p_band, p_type)
    assert np.allclose(p_any, [0.05, 1.0])
    l, _, _ = V.rules(pres, V.Thresholds())
    assert list(l) == [V.LAB_PURE, V.REJECT]  # a filler guess alone does not condemn a likely-pure sample


def test_judge():
    assert V.judge("P", 0, V.LAB_PURE, False, False) == "right"
    assert V.judge("D", 6, V.LAB_WOOD, False, False) == "right"
    assert V.judge("D", 6, V.LABEL, True, False) == "wrong"
    assert V.judge("P", 6, V.LABEL, True, False) == "right"
    assert V.judge("P", 6, V.LABEL, True, True) == "partly"
    assert V.judge("W", 6, V.LAB_PURE, False, False) == "wrong"


def test_screened_skips_rows_without_predictions():
    s = pd.DataFrame({"sample_id": list("abcd"), "cell": ["CP0", "CP6", "CD6", "CW6"], "code": list("PPDW"),
                      "level": [0, 6, 6, 6], "locked": [False, False, False, True]})
    p3 = np.array([[0.9, 0.1, 0, 0], [0, 0, 1, 0], [np.nan] * 4, [0, 0, 1, 0]])
    t3 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
    df = V.screened(s, (p3, np.full((4, 4), np.nan)), (t3, np.full((4, 3), np.nan)), V.Thresholds())
    assert list(df.sample_id) == ["a", "b"]  # c has no prediction, d is locked and unscored
    m = V.summary(df)
    assert m["pure_to_verify"] == 1.0 and m["peanut_labelled"] == 1.0
