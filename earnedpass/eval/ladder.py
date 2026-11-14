"""Stress ladder: one fixed model per sensor set, four splits of rising strictness."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, recall_score

from .. import config as C
from .. import models
from . import splits


def metrics(task, y, pred) -> dict:
    if task == "level":
        return {"mae_pct_points": mean_absolute_error(y, pred)}
    out = {"balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    if task == "detect":
        r = recall_score(y, pred, average=None, labels=[0, 1], zero_division=0)
        out["recall_0pct"], out["recall_blend"] = r[0], r[1]
    return out


HEADLINE = {"detect": "balanced_accuracy", "band": "macro_f1", "level": "mae_pct_points", "type": "macro_f1"}


def run(data, spectra, idx, rungs=C.RUNGS, sensor_sets=tuple(C.SENSOR_SETS), tasks=tuple(C.TASKS), ftir_variant="raw", log=print):
    """Out-of-fold predictions on the rows `idx` for every rung, set and task.

    Returns (rows, oof) where oof[(rung, set, task)] = {"pred", "proba"} arrays
    over all samples (NaN outside `idx`).
    """
    s = data.samples
    n = len(s)
    rows, oof = [], {}
    for rung in rungs:
        fl = splits.folds(s, rung, idx)
        log(f"  rung {rung}: {len(fl)} folds")
        for task in tasks:
            for ss in sensor_sets:
                k = len(models.CLASSES[task]) if task != "level" else 0
                oof[(rung, ss, task)] = {"pred": np.full(n, np.nan), "proba": np.full((n, k), np.nan) if k else None}
        for _, tr, te in fl:
            blocks = models.FoldBlocks(data, spectra, tr, ftir_variant)
            for task in tasks:
                if task == "type" and rung == "4_leave_folder_out":
                    continue  # the held-out filler is never in training: undefined, not a result
                te_t = models.task_rows(s, task, te)
                if not len(te_t):
                    continue  # a 0% cell has no blends to type
                for ss in sensor_sets:
                    pred, proba = models.fit_predict(data, blocks, task, ss, tr, te_t)
                    o = oof[(rung, ss, task)]
                    o["pred"][te_t] = pred
                    if proba is not None:
                        o["proba"][te_t] = proba
        for task in tasks:
            y = models.targets(s, task)
            for ss in sensor_sets:
                pred = oof[(rung, ss, task)]["pred"]
                ok = ~np.isnan(pred)
                if not ok.any():
                    continue
                for name, v in metrics(task, y[ok], pred[ok]).items():
                    rows.append(
                        dict(section="ladder", task=task, sensor_set=ss, rung=rung, mode="-", metric=name, value=float(v),
                             n_samples=int(ok.sum()), n_cells=int(s.cell[ok].nunique()))
                    )
    return rows, oof
