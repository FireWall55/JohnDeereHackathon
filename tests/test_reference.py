import numpy as np

from earnedpass import config as C
from earnedpass import models, reference, verdict
from tests.test_pipeline import synthetic


def test_split_keeps_groups_together_and_uses_only_pure_samples():
    d = synthetic(reps=8)
    is_ref = reference.split(d.samples)
    reference.check_split(d.samples, is_ref)
    assert not is_ref[(d.samples.level > 0).to_numpy()].any()
    for code in C.FOLDERS:
        zero = ((d.samples.code == code) & (d.samples.level == 0)).to_numpy()
        assert 0 < is_ref[zero].sum() < zero.sum()


def test_reference_run_never_screens_reference_or_locked_rows():
    d = synthetic(reps=8)
    out = reference.run(d, models.Spectra(d.X_ftir), log=lambda *_: None)
    s, is_ref = d.samples, out["is_ref"]
    locked = s.locked.to_numpy()
    screened = ~locked & ~is_ref
    for k in ("p1", "p3", "t3"):
        assert np.isfinite(out[k][:, screened]).all(), k
        assert np.isnan(out[k][:, is_ref | locked]).all(), k
    assert np.isnan(out["pl"]).all() and np.isnan(out["el"]).all()  # locked cells untouched
    df = verdict.screened(s, (out["p3"][0], out["pl"][0]), (out["t3"][0], out["tl"][0]), verdict.Thresholds())
    assert not df.row.isin(np.flatnonzero(is_ref)).any()
    assert len(df) == screened.sum()
    assert set(out["new_supplier"]) == set(C.SENSOR_SETS)


def test_reference_folds_hold_out_whole_units():
    d = synthetic(reps=8)
    is_ref = reference.split(d.samples)
    for rung in ("1_sample_id", "3_leave_cell_out", "4_new_supplier"):
        fl = reference.folds(d.samples, is_ref, rung)
        tested = np.concatenate([te for te, _ in fl])
        assert sorted(tested) == sorted(np.flatnonzero(~d.samples.locked.to_numpy() & ~is_ref))
