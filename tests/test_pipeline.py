"""End-to-end on synthetic data shaped like the real thing, including the
locked-cell scoring path that the real data only goes through once."""
import numpy as np
import pandas as pd

from earnedpass import config as C
from earnedpass import decide, models, run_all
from earnedpass.data import Data


def synthetic(reps=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for code in C.FOLDERS:
        for level in C.LEVELS:
            for rep in range(1, reps + 1):
                rows.append(dict(sample_id=f"C{code}{level}S{rep:02d}", code=code, folder=C.FOLDERS[code], level=level, rep=rep,
                                 cell=f"C{code}{level}", band=C.band_of(level), locked=level in C.LOCKED_LEVELS))
    s = pd.DataFrame(rows)
    lv = s.level.to_numpy()[:, None] / 24
    s["L"], s["a"], s["b"] = (50 + 5 * lv[:, 0] + rng.normal(0, 0.3, len(s)), 10 - 2 * lv[:, 0] + rng.normal(0, 0.3, len(s)),
                              20 + rng.normal(0, 0.3, len(s)))
    s["chroma"], s["hue"] = np.hypot(s.a, s.b), np.degrees(np.arctan2(s.b, s.a))
    s["merged_group"] = np.arange(len(s)) // 2
    X_ftir = 1 - 0.1 * lv * rng.random((1, 60)) + rng.normal(0, 0.002, (len(s), 60))
    X_img = lv * rng.normal(0, 1, (1, 19)) + rng.normal(0, 0.2, (len(s), 19))
    return Data(s, X_ftir.astype(np.float32), np.linspace(400, 4000, 60), X_img.astype(np.float32))


def test_pipeline_with_locked_scoring():
    d = synthetic()
    sp = models.Spectra(d.X_ftir)
    quiet = lambda *_: None
    rows, p1, p3, pl = run_all.model_all(d, sp, score_locked=True, log=quiet)
    locked = d.samples.locked.to_numpy()
    assert np.isfinite(p1[:, ~locked]).all() and np.isnan(p1[:, locked]).all()  # open cells only in the ladder
    assert np.isfinite(pl[:, locked]).all() and np.isnan(pl[:, ~locked]).all()
    assert np.allclose(pl[:, locked].sum(-1), 1)
    r = pd.DataFrame(rows)
    assert set(r.rung) == set(C.RUNGS) | {"locked_cells"}
    assert r[(r.rung == "4_leave_folder_out") & (r.task == "type")].empty
    assert (r[r.section == "locked"].n_cells.isin([9])).all()

    probs = {ss: decide.Probs(p1[i], p3[i], pl[i]) for i, ss in enumerate(C.SENSOR_SETS)}
    summary = run_all.decision_summary(d.samples, rows, probs)
    res = decide.evaluate(d.samples, probs[C.SCREEN_SET], decide.Settings())
    assert res.pools == ("dev", "locked") and len(res.df) == len(d.samples)
    assert res.claims["catch"]["cells"]["n"] == 24 and res.claims["reference"]["cells"]["n"] == 3
    assert not summary[C.SCREEN_SET]["replicate_aware"]["licence"]["licensed"]  # three 0% cells cap claim 2 at 36.8%


def test_type_probabilities_cover_held_out_samples_only():
    d = synthetic(reps=6)
    sp = models.Spectra(d.X_ftir)
    locked = d.samples.locked.to_numpy()
    t3, tl = run_all.type_probabilities(d, sp)
    assert t3.shape == (len(C.SENSOR_SETS), len(d.samples), 3)
    assert np.allclose(t3[:, ~locked].sum(-1), 1)  # 0% samples get a filler guess too
    assert np.isnan(t3[:, locked]).all() and np.isnan(tl).all()  # locked cells untouched
    _, tl = run_all.type_probabilities(d, sp, score_locked=True)
    assert np.allclose(tl[:, locked].sum(-1), 1) and np.isnan(tl[:, ~locked]).all()


def test_level_estimates_cover_held_out_samples_only():
    d = synthetic(reps=6)
    sp = models.Spectra(d.X_ftir)
    locked = d.samples.locked.to_numpy()
    e3, el = run_all.level_estimates(d, sp)
    assert e3.shape == (len(C.SENSOR_SETS), len(d.samples))
    assert np.isfinite(e3[:, ~locked]).all() and np.isnan(e3[:, locked]).all() and np.isnan(el).all()
    assert e3[:, ~locked].min() >= 0 and e3[:, ~locked].max() <= 24
    _, el = run_all.level_estimates(d, sp, score_locked=True)
    assert np.isfinite(el[:, locked]).all() and np.isnan(el[:, ~locked]).all()


def test_unscored_locked_pool_is_refused():
    d = synthetic(reps=6)
    rows, p1, p3, pl = run_all.model_all(d, models.Spectra(d.X_ftir), log=lambda *_: None)
    probs = decide.Probs(p1[0], p3[0], pl[0])
    assert not probs.locked_scored
    assert decide.evaluate(d.samples, probs, decide.Settings()).pools == ("dev",)
