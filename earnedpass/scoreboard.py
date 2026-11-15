"""Plain-language performance numbers for the Scoreboard page.

Everything is computed from stored held-out predictions with the same decision
logic the app uses, so the Scoreboard agrees with Inspect and Evidence.
"Right" is scored as on Inspect: HOLD on any adulterated sample is right, HOLD
on pure cinnamon is wrong, PASS is right only on pure cinnamon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

HEADLINE = ["wrong_pass", "wood_not_held", "pure_held", "decided", "decided_right", "range_has_truth"]
LABEL = {
    "wrong_pass": "Adulterated samples passed",
    "wood_not_held": "Wood dust samples not held",
    "pure_held": "Pure samples held",
    "decided": "Samples decided (PASS or HOLD)",
    "decided_right": "Decisions that were right",
    "range_has_truth": "Amount range contains the truth",
    "amount_error": "Amount estimate error (points)",
    "filler_right": "Filler type named correctly",
    "n": "Samples tested",
    "materials": "Materials tested",
}


def _rate(num, den):
    return float(num) / den if den else float("nan")


def from_lanes(df: pd.DataFrame, lane_col="lane") -> dict:
    """Headline numbers from a table with one row per screened sample."""
    lane, level = df[lane_col].to_numpy(), df.level.to_numpy()
    blend, pure = level > 0, level == 0
    decided = lane != "LAB"
    right = ((lane == "PASS") & pure) | ((lane == "HOLD") & blend)
    return {
        "wrong_pass": _rate(((lane == "PASS") & blend).sum(), blend.sum()),
        "wood_not_held": _rate(((lane != "HOLD") & blend & (df.code.to_numpy() == "D")).sum(),
                               (blend & (df.code.to_numpy() == "D")).sum()) if "code" in df else float("nan"),
        "pure_held": _rate(((lane == "HOLD") & pure).sum(), pure.sum()),
        "decided": _rate(decided.sum(), len(df)),
        "decided_right": _rate((right & decided).sum(), decided.sum()),
        "range_has_truth": float(df.covered.mean()) if "covered" in df else float("nan"),
        "n": int(len(df)),
        "materials": int(df.cell.nunique()),
    }


def forced(df: pd.DataFrame) -> dict:
    """The forced-answer baseline on the same samples: top band zero -> PASS, else HOLD."""
    lanes = np.where(df.top_band.to_numpy() == "zero", "PASS", "HOLD")
    out = from_lanes(df.assign(baseline_lane=lanes), "baseline_lane")
    out["range_has_truth"] = float("nan")  # it gives no range
    return out


def amount_error(est: np.ndarray, rows, levels) -> float:
    e = est[rows]
    ok = np.isfinite(e)
    return float(np.abs(e[ok] - levels[rows][ok]).mean()) if ok.any() else float("nan")


def filler_right(probs: np.ndarray, rows, levels, codes) -> float:
    rows = np.asarray(rows)
    rows = rows[(levels[rows] > 0) & np.isfinite(probs[rows]).all(1)]
    if not len(rows):
        return float("nan")
    return float((probs[rows].argmax(1) == codes[rows]).mean())


def easy_vs_fair(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per instrument set: easy (sample-ID split) against fair (leave-one-material-out)."""
    lad = rows[rows.section == "ladder"]
    spec = [("detect", "balanced_accuracy", "Pure vs adulterated (balanced accuracy)"),
            ("band", "macro_f1", "Filler level band (macro-F1)"),
            ("level", "mae_pct_points", "Amount error (points)")]
    out = []
    for ss in C.SENSOR_SETS:
        rec = {"sensor_set": ss}
        for task, metric, name in spec:
            def get(rung):
                v = lad[(lad.task == task) & (lad.sensor_set == ss) & (lad.rung == rung) & (lad.metric == metric)].value
                return float(v.iloc[0]) if len(v) else float("nan")
            rec[f"{name} · easy"] = get("1_sample_id")
            rec[f"{name} · fair"] = get("3_leave_cell_out")
        out.append(rec)
    return pd.DataFrame(out)
