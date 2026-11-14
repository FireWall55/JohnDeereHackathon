"""Data contract: one row per sample ID, labels parsed from the ID only."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

ID_RE = re.compile(r"^C([PWD])(\d+)S(\d+)$")


def parse_id(sample_id: str):
    m = ID_RE.match(str(sample_id).strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def _category_level(raw) -> float:
    """The dataset's own category column, read only to count disagreements."""
    text = str(raw).strip()
    try:
        v = float(text.replace("%", ""))
    except ValueError:
        return float("nan")
    # Stored as a fraction (0.01 to 0.24) in most sheets and as "1%" text in one.
    return round(v * 100, 6) if ("%" not in text and 0 < v < 1) else v


def _one(folder_dir, sub):
    files = sorted((folder_dir / sub).glob("*.xlsx"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one .xlsx in {folder_dir / sub}, found {len(files)}")
    return files[0]


def load_colour(root=C.DATA_ROOT) -> pd.DataFrame:
    rows = []
    for code, folder in C.FOLDERS.items():
        df = pd.read_excel(_one(root / folder, "Colorimetry"))
        cols = {str(c).strip().lower(): c for c in df.columns}
        out = pd.DataFrame(
            {
                "sample_id": df[cols["sample_name"]].astype(str).str.strip(),
                "L": df[cols["l*"]],
                "a": df[cols["a*"]],
                "b": df[cols["b*"]],
                "category_colour": df[cols["category"]].map(_category_level),
            }
        )
        out = out[out.sample_id.map(lambda s: parse_id(s) is not None)]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def load_ftir(root=C.DATA_ROOT):
    ids, mats, cats, wn = [], [], [], None
    for code, folder in C.FOLDERS.items():
        df = pd.read_excel(_one(root / folder, "FTIR"))
        spec_cols = [c for c in df.columns if isinstance(c, (int, float))]
        this_wn = np.array(spec_cols, dtype=float)
        if wn is None:
            wn = this_wn
        elif not np.allclose(wn, this_wn):
            raise ValueError(f"{folder}: FTIR wavenumber axis differs from the other folders")
        cols = {str(c).strip().lower(): c for c in df.columns if isinstance(c, str)}
        sid = df[cols["sample_name"]].astype(str).str.strip()
        keep = sid.map(lambda s: parse_id(s) is not None).to_numpy()
        ids += sid[keep].tolist()
        mats.append(df.loc[keep, spec_cols].to_numpy(dtype=np.float32))
        cats += df.loc[keep, cols["category"]].map(_category_level).tolist()
    return ids, np.vstack(mats), wn, cats


def index_images(root=C.DATA_ROOT) -> dict:
    out = {}
    for folder in C.FOLDERS.values():
        for p in (root / folder / "Images").rglob("*"):
            if p.suffix.lower() in (".jpg", ".jpeg") and parse_id(p.stem):
                if p.stem in out:
                    raise ValueError(f"image ID {p.stem} appears twice")
                out[p.stem] = str(p)
    return out


def build_samples(root=C.DATA_ROOT):
    """Return (samples, X_ftir, wavenumbers, audit). Raises if a contract check fails."""
    colour = load_colour(root)
    f_ids, X, wn, f_cat = load_ftir(root)
    images = index_images(root)

    for name, ids in (("colour", colour.sample_id), ("FTIR", pd.Series(f_ids)), ("image", pd.Series(list(images)))):
        if ids.duplicated().any():
            raise ValueError(f"{name}: duplicated sample IDs {ids[ids.duplicated()].tolist()[:5]}")
    sets = {"colour": set(colour.sample_id), "ftir": set(f_ids), "image": set(images)}
    common = sets["colour"] & sets["ftir"] & sets["image"]
    for name, s in sets.items():
        if s != common:
            raise ValueError(f"{name}: {len(s - common)} IDs missing from another sensor, e.g. {sorted(s - common)[:5]}")

    s = colour.sort_values("sample_id").reset_index(drop=True)
    parsed = s.sample_id.map(parse_id)
    s["code"] = parsed.str[0]
    s["folder"] = s.code.map(C.FOLDERS)
    s["level"] = parsed.str[1].astype(int)
    s["rep"] = parsed.str[2].astype(int)
    s = s.sort_values(["code", "level", "rep"]).reset_index(drop=True)
    s["cell"] = "C" + s.code + s.level.astype(str)
    s["band"] = s.level.map(C.band_of)
    s["chroma"] = np.hypot(s.a, s.b)
    s["hue"] = np.degrees(np.arctan2(s.b, s.a))
    s["img_path"] = s.sample_id.map(images)
    s["locked"] = s.level.isin(C.LOCKED_LEVELS)

    row_of = {sid: i for i, sid in enumerate(f_ids)}
    order = s.sample_id.map(row_of).to_numpy()
    X = X[order]
    s["ftir_row"] = np.arange(len(s))
    f_cat = np.array(f_cat, dtype=float)[order]

    # Contract checks.
    if len(s) != 1950 or s.sample_id.nunique() != 1950:
        raise ValueError(f"expected 1,950 unique sample IDs, got {s.sample_id.nunique()}")
    per_cell = s.groupby("cell").size()
    if len(per_cell) != 39 or not (per_cell == 50).all():
        raise ValueError(f"expected 39 cells of 50, got {len(per_cell)} cells, sizes {sorted(per_cell.unique())}")
    if sorted(s.level.unique()) != C.LEVELS:
        raise ValueError(f"unexpected levels {sorted(s.level.unique())}")
    if not np.isfinite(X).all() or s[["L", "a", "b"]].isna().any().any():
        raise ValueError("missing values in FTIR or colour readings")

    audit = {
        "n_samples": int(len(s)),
        "n_cells": int(len(per_cell)),
        "category_disagrees_colour": s.sample_id[s.category_colour != s.level].tolist(),
        "category_disagrees_ftir": s.sample_id[f_cat != s.level.to_numpy()].tolist(),
        "ftir_shape": list(X.shape),
        "ftir_value_range": [float(X.min()), float(X.max())],
    }
    return s.drop(columns=["category_colour"]), X, wn, audit
