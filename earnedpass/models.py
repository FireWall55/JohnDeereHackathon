"""Per-fold feature blocks and the task models. Everything is fitted on the
training rows of the fold and nothing else."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import color, ftir

CLASSES = {"detect": ["0%", "blend"], "band": C.BANDS, "type": list(C.FOLDERS)}


def targets(samples, task) -> np.ndarray:
    if task == "detect":
        return (samples.level > 0).astype(int).to_numpy()
    if task == "band":
        return samples.band.map(C.BANDS.index).to_numpy()
    if task == "type":
        return samples.code.map(list(C.FOLDERS).index).to_numpy()
    if task == "level":
        return samples.level.to_numpy(float)
    raise ValueError(task)


def task_rows(samples, task, idx) -> np.ndarray:
    """Type is defined on blends only."""
    idx = np.asarray(idx)
    return idx[samples.level.to_numpy()[idx] > 0] if task == "type" else idx


class Spectra:
    """Absorbance and its first derivative, computed once."""

    def __init__(self, X_ftir):
        self.raw = ftir.absorbance(X_ftir)
        self._deriv = None

    def get(self, variant):
        if variant == "raw":
            return self.raw
        if self._deriv is None:
            self._deriv = ftir.first_derivative(self.raw)
        return self._deriv


class FoldBlocks:
    """Scaled feature blocks for every row, with all statistics from `train`.

    colour: 6 features. ftir: 10 PCA scores of absorbance. image: 10 PCA scores
    of the standardised image features. A sensor set is a choice of blocks.
    """

    def __init__(self, data, spectra, train, ftir_variant="raw", seed=C.SEED):
        s = data.samples
        self.train = np.asarray(train)
        self.refs = color.references(s, self.train)
        self.spec = spectra.get(ftir_variant)

        col = color.block(s, self.refs)
        self._col_scaler = StandardScaler().fit(col[self.train])

        self._ftir_pca = PCA(10, random_state=seed).fit(self.spec[self.train])
        f = self._ftir_pca.transform(self.spec)
        self._ftir_scaler = StandardScaler().fit(f[self.train])

        self._img_scaler = StandardScaler().fit(data.X_img[self.train])
        zi = self._img_scaler.transform(data.X_img)
        self._img_pca = PCA(10, random_state=seed).fit(zi[self.train])
        i = self._img_pca.transform(zi)
        self._img_scaler2 = StandardScaler().fit(i[self.train])

        self.blocks = {
            "colour": self._col_scaler.transform(col),
            "ftir": self._ftir_scaler.transform(f),
            "image": self._img_scaler2.transform(i),
        }

    def X(self, sensors) -> np.ndarray:
        return np.hstack([self.blocks[k] for k in sensors])


def fit_predict(data, blocks: FoldBlocks, task, sensor_set, train, test):
    """Fit on `train`, predict `test`. Returns (prediction, probabilities or None)."""
    s = data.samples
    sensors = C.SENSOR_SETS[sensor_set]
    tr, te = task_rows(s, task, train), np.asarray(test)
    y = targets(s, task)
    if task == "level":
        if sensors == ("ftir",):  # PLS on the spectra, as the final step
            m = PLSRegression(8, scale=False).fit(blocks.spec[tr], y[tr])
            pred = np.ravel(m.predict(blocks.spec[te]))
        else:
            X = blocks.X(sensors)
            pred = Ridge(1.0).fit(X[tr], y[tr]).predict(X[te])
        return np.clip(pred, 0, 30), None
    X = blocks.X(sensors)
    clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(X[tr], y[tr])
    proba = np.zeros((len(te), len(CLASSES[task])))
    proba[:, clf.classes_] = clf.predict_proba(X[te])
    return proba.argmax(1), proba
