"""Groups that tie repeat measurements together: twin photos, repeated colour
readings and photo sessions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from . import config as C


def twin_groups(samples: pd.DataFrame, csv=C.TWINS_CSV) -> pd.Series:
    """Identical-photo groups from image_twins.csv (id, root, cell)."""
    t = pd.read_csv(csv).set_index("id")
    missing = set(samples.sample_id) - set(t.index)
    if missing:
        raise ValueError(f"{len(missing)} sample IDs missing from {csv}")
    t = t.loc[samples.sample_id]
    return pd.Series((t.cell + "-t" + t.root.astype(str)).to_numpy(), index=samples.index)


def colour_groups(samples: pd.DataFrame) -> pd.Series:
    """Identical L*, a*, b* readings inside one cell."""
    key = samples[["L", "a", "b"]].round(2).astype(str).agg("/".join, axis=1)
    k = samples.groupby("cell", group_keys=False).apply(
        lambda g: pd.Series(pd.factorize(key[g.index])[0], index=g.index), include_groups=False
    )
    return samples.cell + "-c" + k.loc[samples.index].astype(str)


def shot_times(samples: pd.DataFrame) -> pd.Series:
    out = []
    for p in samples.img_path:
        with Image.open(p) as im:
            ex = im.getexif()
            t = ex.get_ifd(0x8769).get(36867) or ex.get(306)
        out.append(pd.to_datetime(t, format="%Y:%m:%d %H:%M:%S", errors="coerce"))
    return pd.Series(out, index=samples.index)


def merge_groups(*labels: pd.Series) -> pd.Series:
    """Union-find: samples that share any label end up in one group."""
    n = len(labels[0])
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for lab in labels:
        first = {}
        for i, v in enumerate(lab.to_numpy()):
            if v in first:
                parent[find(i)] = find(first[v])
            else:
                first[v] = i
    return pd.Series([find(i) for i in range(n)], index=labels[0].index)


def add_groups(samples: pd.DataFrame) -> pd.DataFrame:
    s = samples.copy()
    s["twin_group"] = twin_groups(s)
    s["colour_group"] = colour_groups(s)
    s["merged_group"] = merge_groups(s.twin_group, s.colour_group)
    s["shot_time"] = shot_times(s)
    s["session"] = s.shot_time.dt.strftime("%Y-%m-%d").fillna("unknown")
    return s
