"""Colour features: L*, a*, b*, chroma, hue and delta-E to a retained reference."""
from __future__ import annotations

import numpy as np

NAMES = ["L", "a", "b", "chroma", "hue", "dE_ref"]


def references(samples, train_idx) -> dict:
    """Mean L*a*b* of the 0% samples the model may see, per folder.

    Stands in for the inspector's retained known-good sample. A folder with no
    0% sample in training (its 0% cell is held out, or the whole folder is)
    falls back to the mean of the other folders' references under key None.
    """
    tr = samples.iloc[train_idx]
    zero = tr[tr.level == 0]
    refs = {code: g[["L", "a", "b"]].mean().to_numpy() for code, g in zero.groupby("code")}
    refs[None] = np.mean(list(refs.values()), axis=0) if refs else tr[["L", "a", "b"]].mean().to_numpy()
    return refs


def block(samples, refs) -> np.ndarray:
    lab = samples[["L", "a", "b"]].to_numpy(float)
    ref = np.vstack([refs.get(c, refs[None]) for c in samples.code])
    dE = np.linalg.norm(lab - ref, axis=1)
    return np.column_stack([lab, samples.chroma, samples.hue, dE])
