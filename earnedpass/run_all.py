"""Run the whole pipeline and write the numbers registry.

    python -m earnedpass.run_all                  # Saturday: open cells only
    python -m earnedpass.run_all --score-locked   # Sunday 8 AM: score the locked cells, once

Every number shown on a slide or in the app's defaults is read from
artifacts/results.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess

import numpy as np

from . import audit, controls, data as data_mod, decide, models, reference
from . import config as C
from .eval import ladder, splits
from .eval.bounds import cp_lower, units_needed
from .eval.conformal import min_alpha

PROBS = C.ARTIFACTS / "band_probs.npz"
PROBS_REF = C.ARTIFACTS / "band_probs_reference.npz"
RESULTS = C.ARTIFACTS / "results.json"


def load_probs(sensor_set) -> decide.Probs:
    z = np.load(PROBS, allow_pickle=False)
    i = list(z["sets"]).index(sensor_set)
    return decide.Probs(z["p1"][i], z["p3"][i], z["pl"][i])


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def decision_summary(samples, rows, probs_by_set) -> dict:
    """Default-setting decision results for every sensor set and both modes."""
    summary = {}
    for ss, probs in probs_by_set.items():
        for mode in decide.MODES:
            res = decide.evaluate(samples, probs, decide.Settings(mode=mode, sensor_set=ss))
            base = dict(section="decision", task="band", sensor_set=ss, rung="3_leave_cell_out", mode=mode,
                        n_samples=res.metrics["n_samples"], n_cells=res.metrics["n_cells"])
            for k, v in res.metrics.items():
                if k not in ("n_samples", "n_cells"):
                    rows.append({**base, "metric": k, "value": float(v)})
            for who, b in (("b1_held_out_cells", res.baseline), ("b1_sample_id_split", res.baseline_sample_split)):
                for k, v in b.items():
                    if k not in ("n_samples", "n_cells"):
                        rows.append({**base, "rung": "1_sample_id" if "sample_id" in who else base["rung"], "mode": "-",
                                     "metric": f"{who}.{k}", "value": float(v), "n_samples": b["n_samples"], "n_cells": b["n_cells"]})
            for claim, counts in res.claims.items():
                for unit, c in counts.items():
                    rows.append({**base, "metric": f"claim_{claim}.{unit}.bound", "value": c["bound"],
                                 "n_samples": c["n"] if unit == "samples" else None, "n_cells": c["n"] if unit == "cells" else None})
            summary.setdefault(ss, {})[mode] = {
                "licence": res.licence, "claims": res.claims, "metrics": res.metrics, "baseline": res.baseline,
                "baseline_sample_split": res.baseline_sample_split,
                "tiles": res.tiles[["code", "level", "PASS", "LAB", "HOLD", "status"]].to_dict("records"),
                "pools": list(res.pools),
            }
        fb = decide.threshold_fallback(samples, probs, C.ALPHA, C.MUST_CATCH)
        summary[ss]["threshold_fallback"] = fb
    return summary


def model_all(d, spectra, score_locked=False, log=print):
    """Ladder on the open cells, band probabilities for every sensor set and,
    if asked, the one-time scoring of the locked cells."""
    s, dev, locked = d.samples, d.dev, d.locked
    log("stress ladder on the open cells...")
    rows, oof = ladder.run(d, spectra, dev, log=log)
    sets = list(C.SENSOR_SETS)
    p1 = np.stack([oof[("1_sample_id", ss, "band")]["proba"] for ss in sets])
    p3 = np.stack([oof[("3_leave_cell_out", ss, "band")]["proba"] for ss in sets])
    pl = np.full_like(p1, np.nan)
    if score_locked:
        log("scoring the locked cells with the model trained on the open cells...")
        blocks = models.FoldBlocks(d, spectra, dev)
        for task in C.TASKS:
            te = models.task_rows(s, task, locked)
            y = models.targets(s, task)
            for i, ss in enumerate(sets):
                pred, proba = models.fit_predict(d, blocks, task, ss, dev, te)
                if task == "band":
                    pl[i][te] = proba
                for name, v in ladder.metrics(task, y[te], pred).items():
                    rows.append(dict(section="locked", task=task, sensor_set=ss, rung="locked_cells", mode="-", metric=name,
                                     value=float(v), n_samples=int(len(te)), n_cells=int(s.cell.iloc[te].nunique())))
    return rows, p1, p3, pl


def type_probabilities(d, spectra, score_locked=False):
    """Filler-type probabilities for every held-out sample, 0% samples included.

    The type model is fitted on blends only, so it always names one of the three
    fillers. t3: each open cell judged by the model that never saw it. tl: the
    locked cells, judged by the model trained on the open cells (NaN until scored).
    """
    s, dev, locked = d.samples, d.dev, d.locked
    sets = list(C.SENSOR_SETS)
    t3 = np.full((len(sets), len(s), len(C.FOLDERS)), np.nan)
    tl = np.full_like(t3, np.nan)
    for _, tr, te in splits.folds(s, "3_leave_cell_out", dev):
        blocks = models.FoldBlocks(d, spectra, tr)
        for i, ss in enumerate(sets):
            t3[i][te] = models.fit_predict(d, blocks, "type", ss, tr, te)[1]
    if score_locked:
        blocks = models.FoldBlocks(d, spectra, dev)
        for i, ss in enumerate(sets):
            tl[i][locked] = models.fit_predict(d, blocks, "type", ss, dev, locked)[1]
    return t3, tl


def level_estimates(d, spectra, score_locked=False):
    """Estimated filler level (% w/w) for every held-out sample.

    e3: each open cell estimated by the model that never saw it (leave-one-cell-out).
    el: the locked cells, by the model trained on the open cells (NaN until scored).
    """
    s, dev, locked = d.samples, d.dev, d.locked
    sets = list(C.SENSOR_SETS)
    e3 = np.full((len(sets), len(s)), np.nan)
    el = np.full_like(e3, np.nan)
    for _, tr, te in splits.folds(s, "3_leave_cell_out", dev):
        blocks = models.FoldBlocks(d, spectra, tr)
        for i, ss in enumerate(sets):
            e3[i][te] = models.fit_predict(d, blocks, "level", ss, tr, te)[0]
    if score_locked:
        blocks = models.FoldBlocks(d, spectra, dev)
        for i, ss in enumerate(sets):
            el[i][locked] = models.fit_predict(d, blocks, "level", ss, dev, locked)[0]
    return np.clip(e3, 0, 24), np.clip(el, 0, 24)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-locked", action="store_true", help="score the 9 locked cells (levels 3, 10, 18). Do this once.")
    ap.add_argument("--force", action="store_true", help="rebuild the data cache, or re-score locked cells already scored")
    args = ap.parse_args(argv)

    data_mod.build(force=args.force and not args.score_locked)
    d = data_mod.get()
    s = d.samples
    spectra = models.Spectra(d.X_ftir)
    dev, locked = d.dev, d.locked
    previous = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    was_scored = previous.get("meta", {}).get("locked_scored_at")
    if args.score_locked and was_scored and not args.force:
        raise SystemExit(f"The locked cells were already scored at {was_scored}. They are meant to be touched once; "
                         "pass --force only if you accept that.")

    rows, p1, p3, pl = model_all(d, spectra, score_locked=bool(args.score_locked or was_scored))
    locked_scored_at = (was_scored or dt.datetime.now().isoformat(timespec="seconds")) if (args.score_locked or was_scored) else None
    sets = list(C.SENSOR_SETS)
    print("filler-type predictions for held-out samples...")
    t3, tl = type_probabilities(d, spectra, score_locked=bool(args.score_locked or was_scored))
    print("filler-level estimates for held-out samples...")
    e3, el = level_estimates(d, spectra, score_locked=bool(args.score_locked or was_scored))
    np.savez_compressed(PROBS, sets=np.array(sets), p1=p1, p3=p3, pl=pl, t3=t3, tl=tl, e3=e3, el=el)

    print("lanes, licence and blind-spot map...")
    decision = decision_summary(s, rows, {ss: decide.Probs(p1[i], p3[i], pl[i]) for i, ss in enumerate(sets)})

    print("supplier-reference mode...")
    ref = reference.run(d, spectra, score_locked=bool(args.score_locked or was_scored))
    np.savez_compressed(PROBS_REF, sets=np.array(sets), **{k: ref[k] for k in ("p1", "p3", "pl", "t3", "tl", "e3", "el", "is_ref")})
    ref_rows = []
    ref_decision = decision_summary(s, ref_rows, {ss: decide.Probs(ref["p1"][i], ref["p3"][i], ref["pl"][i])
                                                   for i, ss in enumerate(sets)})
    for r in ref_rows:
        r["section"] = "reference_decision"
    rows += ref_rows
    reference_summary = {
        "decision": ref_decision,
        "new_supplier": ref["new_supplier"],
        "split": {"reference_samples": {c: int((ref["is_ref"] & (s.code == c).to_numpy()).sum()) for c in C.FOLDERS},
                  "screened_pure_samples": {c: int((~ref["is_ref"] & (s.code == c).to_numpy() & (s.level == 0).to_numpy()).sum())
                                            for c in C.FOLDERS}},
        "caveat": "The reference half and the screened pure half come from the same batch, so pure-cinnamon results "
                  "measure 'same powder as the reference', not a new pure delivery from that supplier.",
    }

    print("controls and Replicate Radar...")
    ctrl = {
        "session_probe": controls.session_probe(d, dev),
        "zero_filler_probe": controls.zero_filler_probe(d, spectra, dev),
        "offset_test": controls.offset_test(d, spectra, dev),
        "band_check": controls.band_check(d, spectra, dev),
    }
    radar = audit.radar(d, spectra)
    load_audit = json.loads((C.ARTIFACTS / "load_audit.json").read_text())

    y = (s.level.to_numpy()[dev] > 0)
    created = dt.datetime.now().isoformat(timespec="seconds")
    sha = _git_sha()
    out = {
        "meta": {
            "created": created,
            "git_sha": sha,
            "model_version": f"{sha or 'nogit'}-{created.replace(':', '').replace('-', '')}",
            "locked_scored_at": locked_scored_at,
            "locked_levels": C.LOCKED_LEVELS,
            "n_open": {"samples": int(len(dev)), "cells": int(s.cell.iloc[dev].nunique())},
            "n_locked": {"samples": int(len(locked)), "cells": int(s.cell.iloc[locked].nunique())},
            "defaults": {"alpha": C.ALPHA, "must_catch": C.MUST_CATCH, "pass_bar": C.PASS_BAR, "screen_set": C.SCREEN_SET, "lab_set": C.LAB_SET},
            "ftir_units_assumed": "transmittance (absorbance = -log10)",
        },
        "floor": {
            "always_adulterated_plain_accuracy_all_39_cells": 36 / 39,
            "always_adulterated_plain_accuracy_open_cells": float(y.mean()),
            "always_adulterated_balanced_accuracy": 0.5,
        },
        "limits": {
            "min_alpha_open_cells": min_alpha(int(s.cell.iloc[dev].nunique())),
            "min_alpha_three_zero_cells": min_alpha(3),
            "worked_bounds": {"1200_of_1200_samples": cp_lower(1200, 1200), "24_of_24_cells": cp_lower(24, 24),
                              "50_of_50_samples": cp_lower(50, 50), "1_of_1_cell": cp_lower(1, 1), "3_of_3_zero_materials": cp_lower(3, 3)},
            "lots_needed": {str(t): units_needed(t) for t in (0.90, 0.95, 0.99)},
        },
        "load_audit": load_audit,
        "radar": radar,
        "controls": ctrl,
        "decision": decision,
        "reference": reference_summary,
        "rows": rows,
    }
    RESULTS.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {RESULTS} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
