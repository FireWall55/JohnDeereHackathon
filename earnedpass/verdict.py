"""The team's verdict rules (Mentor Notes, Sat Nov 14).

For each filler the screen computes the chance it is present:

    chance present = chance of any filler (band model: 1 - P(0% band))
                     x chance it is this filler, given there is one (filler model)

Rules, each threshold adjustable:
- Wood dust above the reject limit (default 2%): REJECT, discard.
- Wood dust above the lab limit (default 1%): LAB, wood risk, confirm before use.
- Peanut shell above its limit (default 10%): LABEL as containing peanut shell.
- Walnut shell above its limit (default 10%): LABEL as containing walnut shell.
  Both can apply; the screen labels both and does not choose between them.
- Nothing above its limit: LAB, verify it is pure.
Wood decides the verdict when it fires; nut labels are still reported alongside.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

REJECT, LAB_WOOD, LABEL, LAB_PURE = "REJECT", "LAB_WOOD", "LABEL", "LAB_PURE"
LANES = [REJECT, LAB_WOOD, LABEL, LAB_PURE]
P, W, D = (list(C.FOLDERS).index(c) for c in ("P", "W", "D"))


@dataclass(frozen=True)
class Thresholds:
    wood_lab: float = 0.01
    wood_reject: float = 0.02
    peanut: float = 0.10
    walnut: float = 0.10


def chances(p_band: np.ndarray, p_type: np.ndarray):
    """(chance of any filler, chance present per filler in folder order P, W, D)."""
    p_any = 1.0 - p_band[:, 0]
    return p_any, p_any[:, None] * p_type


def rules(present: np.ndarray, th: Thresholds):
    """Verdict lane plus the two nut labels for each row of `present` (P, W, D)."""
    wood, peanut, walnut = present[:, D], present[:, P], present[:, W]
    label_p, label_w = peanut > th.peanut, walnut > th.walnut
    lane = np.where(wood > th.wood_reject, REJECT,
                    np.where(wood > th.wood_lab, LAB_WOOD,
                             np.where(label_p | label_w, LABEL, LAB_PURE)))
    return lane, label_p, label_w


def judge(code, level, lane, label_p, label_w) -> str:
    """Was the verdict right for what the sample really is?"""
    if level == 0:
        return "right" if lane == LAB_PURE else "wrong"
    if code == "D":
        return "right" if lane in (REJECT, LAB_WOOD) else "wrong"
    if lane in (REJECT, LAB_WOOD):
        return "wrong"
    own = label_p if code == "P" else label_w
    other = label_w if code == "P" else label_p
    return "right" if own and not other else ("partly" if own else "wrong")


def table(samples: pd.DataFrame, rows, p_band, p_type, th: Thresholds, pool: str) -> pd.DataFrame:
    rows = np.asarray(rows)
    p_any, present = chances(p_band[rows], p_type[rows])
    lane, lp, lw = rules(present, th)
    s = samples.iloc[rows]
    df = pd.DataFrame({
        "row": rows, "sample_id": s.sample_id.to_numpy(), "cell": s.cell.to_numpy(), "code": s.code.to_numpy(),
        "level": s.level.to_numpy(), "pool": pool, "p_any": p_any,
        "p_peanut": present[:, P], "p_walnut": present[:, W], "p_wood": present[:, D],
        "lane": lane, "label_peanut": lp, "label_walnut": lw,
    })
    df["result"] = [judge(c, l, ln, a, b) for c, l, ln, a, b in zip(df.code, df.level, df.lane, df.label_peanut, df.label_walnut)]
    return df


def screened(samples: pd.DataFrame, band: tuple, types: tuple, th: Thresholds) -> pd.DataFrame:
    """Every held-out sample with a verdict. band = (p3, pl), types = (t3, tl).

    Open materials are judged by models that never saw them; locked materials only
    once they have been scored. Rows without predictions (reference samples) are skipped.
    """
    (p3, pl), (t3, tl) = band, types
    locked = samples.locked.to_numpy()
    parts = []
    for pool, pb, pt, mask in (("dev", p3, t3, ~locked), ("locked", pl, tl, locked)):
        ok = mask & np.isfinite(pb).all(1) & np.isfinite(pt).all(1)
        if ok.any():
            parts.append(table(samples, np.flatnonzero(ok), pb, pt, th, pool))
    return pd.concat(parts, ignore_index=True)


def summary(df: pd.DataFrame) -> dict:
    blend, pure = df.level > 0, df.level == 0
    wood, peanut, walnut = blend & (df.code == "D"), blend & (df.code == "P"), blend & (df.code == "W")
    caught = df.lane.isin([REJECT, LAB_WOOD])

    def rate(mask, of):
        return float(mask[of].mean()) if of.any() else float("nan")

    return {
        "wood_let_through": rate(~caught, wood),
        "wood_rejected": rate(df.lane == REJECT, wood),
        "filler_missed": rate(df.lane == LAB_PURE, blend),
        "peanut_labelled": rate(df.label_peanut & ~caught, peanut),
        "walnut_labelled": rate(df.label_walnut & ~caught, walnut),
        "wrong_nut_label": rate((df.code == "P") & df.label_walnut | (df.code == "W") & df.label_peanut, blend & (df.code != "D")),
        "pure_to_verify": rate(df.lane == LAB_PURE, pure),
        "pure_rejected": rate(df.lane == REJECT, pure),
        "right": float((df.result == "right").mean()),
        "n": int(len(df)),
        "materials": int(df.cell.nunique()),
    }
