"""Supplier-reference mode: judge each sample against its supplier's own pure sample.

Each folder stands in for one supplier. Half of its 0% cell (split by merged
group, so no identical photo or colour reading is on both sides) becomes the
supplier's retained reference sample; the other half is screened like any other
delivery. Every feature is the sample's measurement minus the reference mean.

Caveat, stated wherever the results are shown: the reference half and the
screened pure half are the same batch, so pure-cinnamon results here measure
"same powder as the reference", not "a new pure delivery from this supplier".
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import models

SEED_SPLIT = 0


def split(samples, seed=SEED_SPLIT) -> np.ndarray:
    """Boolean mask: True for rows used as a supplier's reference sample."""
    is_ref = np.zeros(len(samples), dtype=bool)
    rng = np.random.default_rng(seed)
    for code in C.FOLDERS:
        rows = np.flatnonzero(((samples.code == code) & (samples.level == 0)).to_numpy())
        groups = np.array(sorted(samples.merged_group.iloc[rows].unique()))
        rng.shuffle(groups)
        ref_groups = set(groups[: len(groups) // 2].tolist())
        is_ref[rows[samples.merged_group.iloc[rows].isin(ref_groups).to_numpy()]] = True
    return is_ref


def check_split(samples, is_ref) -> None:
    g = samples.merged_group.to_numpy()
    zero = (samples.level == 0).to_numpy()
    assert not (is_ref & ~zero).any(), "a reference sample is not 0%"
    for code in C.FOLDERS:
        mine = (samples.code == code).to_numpy() & zero
        assert (is_ref & mine).any() and (~is_ref & mine).any(), f"{code}: a half is empty"
    both = set(g[is_ref]) & set(g[zero & ~is_ref])
    assert not both, f"group {sorted(both)[:3]} is both reference and screened"


class RefBlocks(models.FoldBlocks):
    """Feature blocks as differences from the supplier's reference sample.

    colour: dL*, da*, db*, delta-E (4). ftir: 10 PCA scores of absorbance minus the
    reference spectrum. image: 10 PCA scores of image features minus the reference.
    """

    def __init__(self, data, spectra, train, is_ref, seed=C.SEED):
        s = data.samples
        self.train = np.asarray(train)
        code = s.code.to_numpy()
        lab = s[["L", "a", "b"]].to_numpy(float)
        A = spectra.get("raw")
        img = data.X_img.astype(float)

        def minus_ref(X):
            ref = {c: X[is_ref & (code == c)].mean(0) for c in C.FOLDERS}
            return X - np.vstack([ref[c] for c in code])

        d_lab = minus_ref(lab)
        col = np.column_stack([d_lab, np.linalg.norm(d_lab, axis=1)])
        self.spec = minus_ref(A).astype(np.float32)
        d_img = minus_ref(img)

        self._col_scaler = StandardScaler().fit(col[self.train])
        self._ftir_pca = PCA(10, random_state=seed).fit(self.spec[self.train])
        f = self._ftir_pca.transform(self.spec)
        self._ftir_scaler = StandardScaler().fit(f[self.train])
        self._img_scaler = StandardScaler().fit(d_img[self.train])
        zi = self._img_scaler.transform(d_img)
        self._img_pca = PCA(10, random_state=seed).fit(zi[self.train])
        i = self._img_pca.transform(zi)
        self._img_scaler2 = StandardScaler().fit(i[self.train])
        self.blocks = {
            "colour": self._col_scaler.transform(col),
            "ftir": self._ftir_scaler.transform(f),
            "image": self._img_scaler2.transform(i),
        }


def folds(samples, is_ref, rung, seed=C.SEED):
    """Held-out units over the open, non-reference rows. Reference rows are always
    available for training: they are the supplier's retained sample."""
    open_ = ~samples.locked.to_numpy()
    screened = np.flatnonzero(open_ & ~is_ref)
    refs = np.flatnonzero(is_ref)
    cell = samples.cell.to_numpy()
    out = []
    if rung == "1_sample_id":
        kf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for a, b in kf.split(screened, cell[screened]):
            out.append((screened[b], np.concatenate([screened[a], refs])))
    elif rung == "3_leave_cell_out":
        for c in np.unique(cell[screened]):
            te = screened[cell[screened] == c]
            out.append((te, np.concatenate([screened[cell[screened] != c], refs])))
    elif rung == "4_new_supplier":
        code = samples.code.to_numpy()
        for c in C.FOLDERS:
            te = screened[code[screened] == c]
            tr = np.concatenate([screened[code[screened] != c], refs[code[refs] != c]])
            out.append((te, tr))
    else:
        raise ValueError(rung)
    ids, g = samples.sample_id.to_numpy(), samples.merged_group.to_numpy()
    for te, tr in out:
        assert not set(ids[te]) & set(ids[tr]), f"{rung}: a sample ID sits on both sides"
        if rung != "1_sample_id":
            assert not set(g[te]) & set(g[tr]), f"{rung}: an identical photo or reading sits on both sides"
    return out


def run(d, spectra, score_locked=False, log=print):
    """All held-out predictions in reference mode, shaped like the general mode's.

    Returns arrays over (sensor sets, samples, ...): p1, p3, pl (band), t3, tl
    (filler type), e3, el (level), plus is_ref and the new-supplier test.
    Reference rows are NaN everywhere: they are never screened.
    """
    s = d.samples
    n, sets = len(s), list(C.SENSOR_SETS)
    is_ref = split(s)
    check_split(s, is_ref)
    nb, nt = len(C.BANDS), len(C.FOLDERS)
    p1, p3, pl = (np.full((len(sets), n, nb), np.nan) for _ in range(3))
    t3, tl = (np.full((len(sets), n, nt), np.nan) for _ in range(2))
    e3, el = (np.full((len(sets), n), np.nan) for _ in range(2))

    def predict(te, tr, band_out, type_out=None, level_out=None):
        blocks = RefBlocks(d, spectra, tr, is_ref)
        for i, ss in enumerate(sets):
            band_out[i][te] = models.fit_predict(d, blocks, "band", ss, tr, te)[1]
            if type_out is not None:
                type_out[i][te] = models.fit_predict(d, blocks, "type", ss, tr, te)[1]
            if level_out is not None:
                level_out[i][te] = models.fit_predict(d, blocks, "level", ss, tr, te)[0]

    log("  reference mode: sample-ID split...")
    for te, tr in folds(s, is_ref, "1_sample_id"):
        predict(te, tr, p1)
    log("  reference mode: leave-one-cell-out...")
    for te, tr in folds(s, is_ref, "3_leave_cell_out"):
        predict(te, tr, p3, t3, e3)
    if score_locked:
        log("  reference mode: scoring the locked cells...")
        tr = np.flatnonzero(~s.locked.to_numpy())
        predict(np.flatnonzero(s.locked.to_numpy()), tr, pl, tl, el)

    log("  reference mode: new-supplier test...")
    b4 = np.full((len(sets), n, nb), np.nan)
    for te, tr in folds(s, is_ref, "4_new_supplier"):
        predict(te, tr, b4)
    lvl = s.level.to_numpy()
    new_supplier = {}
    for i, ss in enumerate(sets):
        ok = np.isfinite(b4[i]).all(1)
        top = b4[i].argmax(1)
        blend, pure = ok & (lvl > 0), ok & (lvl == 0)
        new_supplier[ss] = {
            "blends_flagged": float((top[blend] > 0).mean()),
            "pure_recognised": float((top[pure] == 0).mean()),
            "by_supplier": {c: {"blends_flagged": float((top[blend & (s.code == c).to_numpy()] > 0).mean()),
                                "pure_recognised": float((top[pure & (s.code == c).to_numpy()] == 0).mean())}
                            for c in C.FOLDERS},
        }
    return dict(p1=p1, p3=p3, pl=pl, t3=np.clip(t3, 0, 1), tl=tl, e3=np.clip(e3, 0, 24), el=np.clip(el, 0, 24),
                is_ref=is_ref, new_supplier=new_supplier)
