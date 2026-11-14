import numpy as np
import pandas as pd

from earnedpass import config as C
from earnedpass import decide


def test_candidate_lanes():
    #                zero  low   mid   high
    sets = np.array([[1, 0, 0, 0],   # only zero -> PASS candidate
                     [0, 0, 1, 1],   # inside {mid, high} -> HOLD
                     [0, 0, 0, 1],   # HOLD
                     [0, 1, 1, 0],   # straddles the must-catch level -> LAB
                     [1, 1, 0, 0],   # LAB
                     [0, 0, 0, 0],   # empty -> LAB
                     [1, 1, 1, 1]], dtype=bool)
    assert decide.candidate_lanes(sets, 6).tolist() == ["PASS", "HOLD", "HOLD", "LAB", "LAB", "LAB", "LAB"]
    assert decide.candidate_lanes(sets, 1)[3] == "HOLD"  # low counts once the must-catch level is 1%
    assert decide.candidate_lanes(sets, 15)[1] == "LAB"


def _perfect():
    """A screen that is always right and certain, on the real cell layout."""
    rows = [dict(sample_id=f"C{c}{l}S{r:02d}", code=c, level=l, cell=f"C{c}{l}", band=C.band_of(l), locked=l in C.LOCKED_LEVELS)
            for c in C.FOLDERS for l in C.LEVELS for r in range(1, 51)]
    s = pd.DataFrame(rows)
    p = np.full((len(s), 4), 0.001)
    p[np.arange(len(s)), s.band.map(C.BANDS.index)] = 0.997
    return s, decide.Probs(p, p, np.full_like(p, np.nan))


def test_perfect_screen_is_licensed_only_when_samples_count():
    s, probs = _perfect()
    a = decide.evaluate(s, probs, decide.Settings(mode="as_labelled"))
    r = decide.evaluate(s, probs, decide.Settings(mode="replicate_aware"))
    assert a.metrics["candidate_wrong_auto_decisions"] == r.metrics["candidate_wrong_auto_decisions"] == 0
    # Same predictions, two ways of counting.
    assert a.claims["catch"]["samples"] == {"k": 900, "n": 900, "bound": a.claims["catch"]["samples"]["bound"]}
    assert r.claims["catch"]["cells"]["n"] == 18 and r.claims["reference"]["cells"]["n"] == 3
    assert a.licence["licensed"] and (a.df.lane[a.df.level == 0] == "PASS").all()
    # Three 0% materials certify at most 36.8%, so PASS stays locked whatever the model does.
    assert abs(r.licence["reference_bound"] - 0.368) < 1e-3
    assert not r.licence["licensed"] and (r.df.lane != "PASS").all()
    assert (r.df.lane[r.df.level >= 6] == "HOLD").all()
    assert "not licensed" in decide.reason(r.df[r.df.level == 0].iloc[0], r)


def test_wrong_pass_breaks_the_catch_claim():
    s, probs = _perfect()
    bad = (s.cell == "CP6").to_numpy()
    probs.p3[bad] = [0.997, 0.001, 0.001, 0.001]  # a 6% cell that looks like 0%
    a = decide.evaluate(s, probs, decide.Settings(mode="as_labelled"))
    assert a.claims["catch"]["samples"]["k"] == 850 and a.claims["catch"]["cells"]["k"] == 17
    t = a.tiles.set_index("cell")
    assert t.loc["CP6", "status"] == "wrong" and t.loc["CP8", "status"] == "right"
    assert a.baseline["wrong_auto_decisions"] == 50
