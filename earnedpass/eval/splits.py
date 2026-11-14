"""The four split rungs, from the track rule to leave-one-folder-out.

Every fold is a pair of row positions into the full sample table. Because the
table has one row per sample ID, all three sensors of a sample always land on
the same side; `check` asserts it, plus the rung's own grouping rule.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .. import config as C

GROUP_RULE = {"1_sample_id": "sample_id", "2_twin_aware": "merged_group", "3_leave_cell_out": "cell", "4_leave_folder_out": "code"}


def check(samples, train, test, rung) -> None:
    col = GROUP_RULE[rung]
    ids = samples.sample_id.to_numpy()
    assert not set(ids[train]) & set(ids[test]), f"{rung}: a sample ID sits on both sides"
    g = samples[col].to_numpy()
    both = set(g[train]) & set(g[test])
    assert not both, f"{rung}: {col} {sorted(both)[:3]} sits on both sides"


def folds(samples, rung, idx=None, seed=C.SEED):
    """Return [(name, train_idx, test_idx)] over the rows in `idx` (default: all)."""
    idx = np.arange(len(samples)) if idx is None else np.asarray(idx)
    s = samples.iloc[idx]
    cell = s.cell.to_numpy()
    out = []
    if rung == "1_sample_id":
        kf = StratifiedKFold(5, shuffle=True, random_state=seed)
        out = [(f"fold{i}", idx[a], idx[b]) for i, (a, b) in enumerate(kf.split(idx, cell))]
    elif rung == "2_twin_aware":
        kf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        out = [(f"fold{i}", idx[a], idx[b]) for i, (a, b) in enumerate(kf.split(idx, cell, s.merged_group.to_numpy()))]
    elif rung == "3_leave_cell_out":
        out = [(c, idx[cell != c], idx[cell == c]) for c in sorted(set(cell), key=lambda c: (c[1], int(c[2:])))]
    elif rung == "4_leave_folder_out":
        code = s.code.to_numpy()
        out = [(c, idx[code != c], idx[code == c]) for c in C.FOLDERS]
    else:
        raise ValueError(rung)
    for _, tr, te in out:
        check(samples, tr, te, rung)
    return out
