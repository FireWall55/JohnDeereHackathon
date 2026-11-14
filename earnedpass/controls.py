"""Controls that test what the models are really reading."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import models
from .eval import ladder, splits

BAND_CHECK = {"1670": (1670, 1670), "1600": (1600, 1600), "1510": (1510, 1510), "1030-1050": (1030, 1050), "3300": (3300, 3300)}


def session_probe(data, idx) -> dict:
    """Can image features tell the shoot day as well as they tell the filler?"""
    s = data.samples
    out = {}
    for name, y in (("session", s.session.to_numpy()), ("filler", s.code.to_numpy())):
        pred = np.empty(len(s), dtype=object)
        for _, tr, te in splits.folds(s, "2_twin_aware", idx):
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
            pred[te] = m.fit(data.X_img[tr], y[tr]).predict(data.X_img[te])
        out[name] = {"balanced_accuracy": float(balanced_accuracy_score(y[idx], pred[idx])),
                     "classes": int(len(set(y[idx]))), "chance": 1 / len(set(y[idx]))}
    return out


def zero_filler_probe(data, spectra, idx) -> dict:
    """Train filler type on blends, apply it to the 0% samples.

    If every 0% sample is given its own folder's filler, the type model is
    reading the folder, not the filler.
    """
    s = data.samples
    zeros = np.flatnonzero((s.level == 0).to_numpy())
    blocks = models.FoldBlocks(data, spectra, idx)  # scalers and references from the open cells
    y = models.targets(s, "type")
    out = {}
    for ss in C.SENSOR_SETS:
        pred, _ = models.fit_predict(data, blocks, "type", ss, idx, zeros)  # fitted on blends only
        out[ss] = {"own_folder_share": float((pred == y[zeros]).mean()), "n_samples": int(len(zeros)), "chance": 1 / 3}
    return out


def offset_test(data, spectra, idx, log=print) -> dict:
    """Level error on raw and first-derivative FTIR, by folder."""
    s = data.samples
    out = {}
    for variant in ("raw", "derivative"):
        _, oof = ladder.run(data, spectra, idx, rungs=("1_sample_id", "3_leave_cell_out"), sensor_sets=("ftir",),
                            tasks=("level",), ftir_variant=variant, log=lambda *_: None)
        for rung in ("1_sample_id", "3_leave_cell_out"):
            pred = oof[(rung, "ftir", "level")]["pred"]
            for code in C.FOLDERS:
                m = idx[(s.code.to_numpy()[idx] == code)]
                out.setdefault(code, {}).setdefault(rung, {})[variant] = float(mean_absolute_error(s.level.to_numpy()[m], pred[m]))
    return out


def band_check(data, spectra, idx) -> dict:
    """Cell-mean absorbance against level at the bands named in the Plan tab."""
    s, wn = data.samples.iloc[idx], data.wavenumbers
    A = spectra.raw[idx]
    out = {}
    for name, (lo, hi) in BAND_CHECK.items():
        cols = [int(np.abs(wn - lo).argmin())] if lo == hi else np.flatnonzero((wn >= lo) & (wn <= hi)).tolist()
        v = A[:, cols].mean(1)
        out[name] = {code: {int(l): float(v[((s.code == code) & (s.level == l)).to_numpy()].mean())
                            for l in sorted(s.level[s.code == code].unique())} for code in C.FOLDERS}
    return out
