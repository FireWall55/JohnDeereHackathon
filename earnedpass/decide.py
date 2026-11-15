"""Decision logic: conformal band sets, the three lanes and the PASS licence.

PASS is the only lane that removes a lab check, so it is the only lane that has
to be earned. Everything here works from stored band probabilities, so the app
can redo it instantly when the QC manager changes a setting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .eval import conformal
from .eval.bounds import cp_lower

MODES = {"as_labelled": "As labelled (every sample counts)", "replicate_aware": "Replicate-aware (every cell counts once)"}
MUST_CATCH_OPTIONS = [1, 6, 15]  # band edges


@dataclass(frozen=True)
class Settings:
    mode: str = "replicate_aware"
    alpha: float = C.ALPHA
    must_catch: int = C.MUST_CATCH
    bar: float = C.PASS_BAR
    sensor_set: str = C.SCREEN_SET
    wood_hold: float | None = None  # HOLD when filler is possible and P(wood) is above this; None = off


@dataclass
class Probs:
    """Band probabilities for one sensor set, one row per sample.

    p1: sample-ID split (rung 1), out of fold.  p3: leave-one-cell-out (rung 3).
    pl: the locked cells, from the model trained on the 30 open cells (NaN until
    the one-time Sunday scoring).
    """

    p1: np.ndarray
    p3: np.ndarray
    pl: np.ndarray
    w3: np.ndarray | None = None  # P(wood dust) from the filler model, judged like p3
    wl: np.ndarray | None = None  # the same for the locked cells

    @property
    def locked_scored(self) -> bool:
        return bool(np.isfinite(self.pl).any())


def _band_index(samples) -> np.ndarray:
    return samples.band.map(C.BANDS.index).to_numpy()


def _scored(p: np.ndarray) -> np.ndarray:
    """Rows that have a prediction. Reference samples in supplier-reference mode have none."""
    return np.isfinite(p).all(1)


def calibrate(samples, probs: Probs, mode, alpha, exclude_cell=None):
    """Return (q, n_units). Calibration never uses the cell being judged."""
    y = _band_index(samples)
    m = ~samples.locked.to_numpy() & _scored(probs.p1 if mode == "as_labelled" else probs.p3)
    if exclude_cell is not None:
        m &= (samples.cell != exclude_cell).to_numpy()
    if mode == "as_labelled":
        sc = conformal.scores(probs.p1[m], y[m])
    elif mode == "replicate_aware":
        sc = conformal.cell_scores(conformal.scores(probs.p3[m], y[m]), samples.cell[m]).to_numpy()
    else:
        raise ValueError(mode)
    return conformal.threshold(sc, alpha), len(sc)


def hold_bands(must_catch) -> list:
    return [b for b in C.BANDS if b != "zero" and C.BAND_FLOOR[b] >= must_catch]


def candidate_lanes(sets: np.ndarray, must_catch) -> np.ndarray:
    """PASS here is a candidate: it still needs the licence."""
    hb = np.array([b in hold_bands(must_catch) for b in C.BANDS])
    any_kept = sets.any(1)
    hold = any_kept & ~(sets & ~hb).any(1)
    passed = sets[:, 0] & (sets.sum(1) == 1)
    return np.where(hold, "HOLD", np.where(passed, "PASS", "LAB"))


def screen(samples, probs: Probs, st: Settings, pool="dev") -> pd.DataFrame:
    """Band set and candidate lane for every held-out sample in the pool.

    dev: each open cell judged by the model that never saw it (rung 3).
    locked: the locked cells, judged by the model trained on the open cells.
    """
    locked = samples.locked.to_numpy()
    if pool == "dev":
        rows, p = np.flatnonzero(~locked & _scored(probs.p3)), probs.p3
    elif pool == "locked":
        rows, p = np.flatnonzero(locked & _scored(probs.pl)), probs.pl
        if not probs.locked_scored:
            raise RuntimeError("the locked cells have not been scored yet")
    else:
        raise ValueError(pool)
    cells = samples.cell.to_numpy()[rows]
    q = np.empty(len(rows))
    n_units = np.empty(len(rows), dtype=int)
    if pool == "dev":
        for c in np.unique(cells):
            q[cells == c], n_units[cells == c] = calibrate(samples, probs, st.mode, st.alpha, exclude_cell=c)
    else:
        q[:], n_units[:] = calibrate(samples, probs, st.mode, st.alpha)
    pr = p[rows]
    sets = (1.0 - pr) <= q[:, None] + 1e-12
    hb = [C.BANDS.index(b) for b in hold_bands(st.must_catch)]
    df = samples.iloc[rows][["sample_id", "code", "level", "band", "cell"]].reset_index(drop=True)
    df["row"] = rows
    df["pool"] = pool
    for j, b in enumerate(C.BANDS):
        df[f"p_{b}"] = pr[:, j]
        df[f"in_{b}"] = sets[:, j]
    df["set"] = [" / ".join(b for b, k in zip(C.BANDS, r) if k) or "(empty)" for r in sets]
    df["top_band"] = np.array(C.BANDS)[pr.argmax(1)]
    df["top_p"] = pr.max(1)
    df["p_must_catch"] = pr[:, hb].sum(1)
    df["covered"] = sets[np.arange(len(rows)), _band_index(samples)[rows]]
    df["q"] = q
    df["n_units"] = n_units
    df["candidate"] = candidate_lanes(sets, st.must_catch)
    return df


def claims(df: pd.DataFrame, must_catch) -> dict:
    """The two claims behind a PASS, each counted in samples and in cells.

    Claim 1 (catch): blends at or above the must-catch level are not passed.
      A held-out cell succeeds if at most 5% of its samples are PASS candidates.
    Claim 2 (reference): genuine cinnamon from a new source is recognised.
      A held-out 0% cell succeeds if at least 90% of its samples get the set {zero}.
    """
    is_pass = df.candidate == "PASS"
    out = {}
    c1 = df[df.level >= must_catch]
    cell_ok = c1.assign(p=is_pass[c1.index]).groupby("cell").p.mean() <= 0.05
    out["catch"] = {
        "samples": {"k": int((~is_pass[c1.index]).sum()), "n": int(len(c1))},
        "cells": {"k": int(cell_ok.sum()), "n": int(len(cell_ok))},
    }
    c2 = df[df.level == 0]
    cell_ok = c2.assign(p=is_pass[c2.index]).groupby("cell").p.mean() >= 0.90
    out["reference"] = {
        "samples": {"k": int(is_pass[c2.index].sum()), "n": int(len(c2))},
        "cells": {"k": int(cell_ok.sum()), "n": int(len(cell_ok))},
    }
    for claim in out.values():
        for count in claim.values():
            count["bound"] = cp_lower(count["k"], count["n"])
    return out


COUNTING = {"as_labelled": "samples", "replicate_aware": "cells"}


def licence(cl: dict, mode, bar) -> dict:
    unit = COUNTING[mode]
    b1, b2 = cl["catch"][unit]["bound"], cl["reference"][unit]["bound"]
    n2 = cl["reference"][unit]["n"]
    return {
        "licensed": bool(b1 >= bar and b2 >= bar),
        "unit": unit,
        "catch_bound": b1,
        "reference_bound": b2,
        "bar": bar,
        # The best claim 2 could reach if every 0% unit succeeded.
        "reference_ceiling": cp_lower(n2, n2),
    }


def metrics(df: pd.DataFrame, alpha) -> dict:
    blend, zero = df.level > 0, df.level == 0
    auto = df.lane != "LAB"
    wrong = ((df.lane == "PASS") & blend) | ((df.lane == "HOLD") & zero)
    cand_wrong = ((df.candidate == "PASS") & blend) | ((df.candidate == "HOLD") & zero)
    return {
        "wrong_pass_rate": float(((df.lane == "PASS") & blend).sum() / max(blend.sum(), 1)),
        "candidate_wrong_pass_rate": float(((df.candidate == "PASS") & blend).sum() / max(blend.sum(), 1)),
        "wrong_hold_rate": float(((df.lane == "HOLD") & zero).sum() / max(zero.sum(), 1)),
        "coverage": float(auto.mean()),
        "candidate_coverage": float((df.candidate != "LAB").mean()),
        "wrong_auto_decisions": int(wrong.sum()),
        "candidate_wrong_auto_decisions": int(cand_wrong.sum()),
        "auto_decisions": int(auto.sum()),
        "lab_share": float((df.lane == "LAB").mean()),
        "hold_share": float((df.lane == "HOLD").mean()),
        "pass_share": float((df.lane == "PASS").mean()),
        "set_coverage": float(df.covered.mean()),
        "stated_coverage": float(1 - alpha),
        "confidence_gap": float(abs((1 - alpha) - df.covered.mean())),
        "mean_set_size": float(df[[f"in_{b}" for b in C.BANDS]].sum(axis=1).mean()),
        "n_samples": int(len(df)),
        "n_cells": int(df.cell.nunique()),
    }


def forced_baseline(samples, p: np.ndarray, rows) -> dict:
    """B1: the same model, forced to call every sample (0% band = PASS, else HOLD)."""
    y = _band_index(samples)[rows]
    pr = p[rows]
    call = pr.argmax(1)
    blend, zero = y > 0, y == 0
    wrong_pass, wrong_hold = (call == 0) & blend, (call > 0) & zero
    acc = float((call == y).mean())
    return {
        "wrong_pass_rate": float(wrong_pass.sum() / max(blend.sum(), 1)),
        "wrong_hold_rate": float(wrong_hold.sum() / max(zero.sum(), 1)),
        "wrong_auto_decisions": int(wrong_pass.sum() + wrong_hold.sum()),
        "auto_decisions": int(len(rows)),
        "coverage": 1.0,
        "band_accuracy": acc,
        "stated_confidence": float(pr.max(1).mean()),
        "confidence_gap": float(abs(pr.max(1).mean() - acc)),
        "n_samples": int(len(rows)),
        "n_cells": int(samples.cell.iloc[rows].nunique()),
    }


def tiles(df: pd.DataFrame) -> pd.DataFrame:
    """Blind-spot map: one tile per held-out cell, before the licence is applied."""
    g = df.groupby(["code", "level", "cell"])
    t = g.candidate.value_counts().unstack(fill_value=0).reindex(columns=["PASS", "LAB", "HOLD"], fill_value=0).reset_index()
    n = t[["PASS", "LAB", "HOLD"]].sum(axis=1)
    wrong = np.where(t.level > 0, t.PASS, t.HOLD)
    t["wrong"] = wrong
    t["status"] = np.where(wrong / n > 0.05, "wrong", np.where(t.LAB / n >= 0.5, "abstains", "right"))
    return t


@dataclass
class Result:
    settings: Settings
    df: pd.DataFrame  # one row per held-out sample, with the final lane
    claims: dict
    licence: dict
    metrics: dict
    baseline: dict  # B1 on the same held-out cells
    baseline_sample_split: dict  # B1 on the sample-ID split, as most entries would score it
    tiles: pd.DataFrame
    pools: tuple


def evaluate(samples, probs: Probs, st: Settings) -> Result:
    pools = ("dev", "locked") if probs.locked_scored else ("dev",)
    df = pd.concat([screen(samples, probs, st, p) for p in pools], ignore_index=True)
    cl = claims(df, st.must_catch)
    lic = licence(cl, st.mode, st.bar)
    df["lane"] = np.where((df.candidate == "PASS") & (not lic["licensed"]), "LAB", df.candidate)
    df["p_wood"] = np.nan
    df["wood_hold"] = False
    if probs.w3 is not None:
        wl = probs.wl if probs.wl is not None else np.full_like(probs.w3, np.nan)
        df["p_wood"] = np.where(df.pool == "dev", probs.w3[df.row], wl[df.row])
        if st.wood_hold is not None:
            # Only where filler is still possible: a shortlist of exactly {0%} means no filler is
            # suspected. An empty shortlist (nothing fits) counts as possible filler.
            filler_possible = ~(df.in_zero & ~df.in_low & ~df.in_mid & ~df.in_high)
            df["wood_hold"] = (filler_possible & (df.p_wood > st.wood_hold)).to_numpy()
            df["lane"] = np.where(df.wood_hold, "HOLD", df.lane)
    held = np.where(df.pool.to_numpy()[:, None] == "dev", probs.p3[df.row], probs.pl[df.row])
    p_held = np.full_like(probs.p3, np.nan)
    p_held[df.row] = held
    dev = np.flatnonzero(~samples.locked.to_numpy() & _scored(probs.p1))
    return Result(st, df, cl, lic, metrics(df, st.alpha), forced_baseline(samples, p_held, df.row.to_numpy()),
                  forced_baseline(samples, probs.p1, dev), tiles(df), pools)


def reason(row, res: Result) -> str:
    """One sentence the inspector can read out."""
    st, lic = res.settings, res.licence
    unit = "samples" if st.mode == "as_labelled" else "independent cells"
    kept = row["set"]
    if row.get("wood_hold", False):
        return (f"Wood dust cannot be ruled out: the filler model gives it {row.p_wood:.0%}, above the "
                f"{st.wood_hold:.0%} limit. Wood is never accepted, so the lot is held for lab confirmation.")
    if row.lane == "HOLD":
        return (f"Every band the screen cannot rule out ({kept}) is at or above the {st.must_catch}% must-catch level: "
                "strong suspicion, confirm at the lab first.")
    if row.lane == "PASS":
        return (f"Only the 0% band is left, and PASS is licensed: both claims are certified at or above "
                f"{st.bar:.0%} on held-out {unit}.")
    if row.candidate == "PASS":
        short = "reference" if lic["reference_bound"] < st.bar else "catch"
        b = lic[f"{short}_bound"]
        return (f"Only the 0% band is left, but PASS is not licensed: the {short} claim is certified at "
                f"{b:.1%}, below the {st.bar:.0%} bar. Routine lab confirmation.")
    if kept == "(empty)":
        return "No band fits this reading at the chosen confidence: unlike anything in calibration. Routine lab confirmation."
    return f"The screen cannot narrow this below {kept}. Routine lab confirmation."


def threshold_fallback(samples, probs: Probs, alpha, must_catch) -> dict:
    """Fallback if conformal slips: abstain when the top probability is below tau.

    tau is chosen on the rung-3 out-of-fold predictions and scored on the same
    predictions, so its numbers are optimistic. Reported for completeness only.
    """
    dev = np.flatnonzero(~samples.locked.to_numpy() & _scored(probs.p3))
    pr, y = probs.p3[dev], _band_index(samples)[dev]
    top, call = pr.max(1), pr.argmax(1)
    hb = [C.BANDS.index(b) for b in hold_bands(must_catch)]
    is_pass, is_hold = call == 0, np.isin(call, hb)
    wrong = (is_pass & (y > 0)) | (is_hold & (y == 0))
    best = {"tau": None, "coverage": 0.0, "wrong_auto_rate": 0.0, "auto_decisions": 0}
    for tau in np.round(np.arange(0.5, 1.0001, 0.01), 2):
        auto = (top >= tau) & (is_pass | is_hold)
        if auto.sum() and wrong[auto].mean() <= alpha:
            best = {"tau": float(tau), "coverage": float(auto.mean()), "wrong_auto_rate": float(wrong[auto].mean()),
                    "auto_decisions": int(auto.sum())}
            break
    return best
