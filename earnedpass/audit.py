"""Replicate Radar: nominal counts against independent counts, per sensor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C


def ftir_structure(samples, A) -> dict:
    """How many distinct mean spectra the 39 cells hold.

    Two cells count as one spectrum when the gap between their mean spectra is
    what repeat noise alone would give: mean squared gap / (2 * se^2) near 1.
    """
    cells = samples.cell.to_numpy()
    means = pd.DataFrame(A).groupby(cells).mean()
    resid = A - means.loc[cells].to_numpy()
    se2 = resid.var() / 50
    lag1 = float((resid[:, 1:] * resid[:, :-1]).mean() / resid.var())
    blocks = {}
    for code in C.FOLDERS:
        names = [f"C{code}{l}" for l in C.LEVELS]
        M = means.loc[names].to_numpy()
        ratio = ((M[:, None, :] - M[None, :, :]) ** 2).mean(-1) / (2 * se2)
        same = ratio < 1.5
        groups, seen = [], set()
        for i, l in enumerate(C.LEVELS):
            if l in seen:
                continue
            g = [C.LEVELS[j] for j in np.flatnonzero(same[i])]
            seen.update(g)
            groups.append(g)
        blocks[code] = {"groups": groups, "noise_ratio": np.round(ratio, 2).tolist()}
    return {
        "within_cell_sd": float(resid.std()),
        "between_cell_sd": float(means.to_numpy().std(0).mean()),
        "residual_lag1_autocorrelation": lag1,
        "blocks": blocks,
        "distinct_mean_spectra": int(sum(len(b["groups"]) for b in blocks.values())),
    }


def twin_pair(samples, thumbs_dir):
    """The largest group of identical photos, and how different two of them are."""
    sizes = samples.groupby("twin_group").size()
    g = sizes.idxmax()
    ids = samples.sample_id[samples.twin_group == g].tolist()
    a, b = (np.asarray(Image.open(thumbs_dir / f"{i}.jpg").convert("RGB"), dtype=float) for i in ids[:2])
    diff = float(np.abs(a - b).mean()) if a.shape == b.shape else float("nan")
    return {"group": g, "ids": ids, "size": int(sizes.max()), "pair": ids[:2], "mean_abs_pixel_difference": diff}


def radar(data, spectra, thumbs_dir=C.THUMBS) -> dict:
    s = data.samples
    span = s.groupby("cell").shot_time.agg(lambda t: (t.max() - t.min()).total_seconds())
    fs = ftir_structure(s, spectra.raw)
    per_cell_twins = s.groupby("cell").twin_group.nunique()
    per_cell_colour = s.groupby("cell").colour_group.nunique()
    return {
        "counts": [
            {"sensor": "Photos", "nominal": len(s), "distinct": int(s.twin_group.nunique()), "cells": 39,
             "strict": int(s.session.nunique()), "distinct_what": "distinct photos", "strict_what": "photo sessions (days)"},
            {"sensor": "Colour", "nominal": len(s), "distinct": int(s.colour_group.nunique()), "cells": 39,
             "strict": None, "distinct_what": "distinct L*a*b* readings", "strict_what": None},
            {"sensor": "FTIR", "nominal": len(s), "distinct": len(s), "cells": 39,
             "strict": fs["distinct_mean_spectra"], "distinct_what": "stored spectra (each is its cell mean plus noise)",
             "strict_what": "distinct mean spectra"},
        ],
        "photos": {
            "distinct_per_cell_median": float(per_cell_twins.median()),
            "distinct_per_cell_min": int(per_cell_twins.min()),
            "cell_shoot_span_seconds_median": float(span.median()),
            "cells_spanning_two_days": s.groupby("cell").session.nunique().loc[lambda x: x > 1].index.tolist(),
            "sessions_by_folder": {c: sorted(g.session.unique()) for c, g in s.groupby("code")},
            "twin_pair": twin_pair(s, thumbs_dir),
        },
        "colour": {
            "distinct_per_cell_min": int(per_cell_colour.min()),
            "cells_with_repeats": int((per_cell_colour < 50).sum()),
            "zero_reference_Lab": {c: g[["L", "a", "b"]].mean().round(1).tolist() for c, g in s[s.level == 0].groupby("code")},
            "within_cell_sd_L": float(s.groupby("cell").L.std().median()),
        },
        "ftir": fs,
    }
