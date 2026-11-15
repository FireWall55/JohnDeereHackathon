"""Final models for screening new samples: colorimeter + photo, supplier reference.

    python -m earnedpass.deploy        # rebuild models/screen_model.pkl from the dataset

The fair-test pipeline (run_all) only replays held-out predictions for dataset
samples. This module trains the three models once on every open material and
saves them, with the scaling and compression steps, so the app can screen a
sample nobody has seen: a photo and a colorimeter reading, compared with that
supplier's reference.

A supplier's reference is the average of one or more photos and one or more
colour readings of its pure cinnamon. More photos give a steadier average; the
models themselves are not retrained.

The locked materials (levels 3, 10, 18) are never used, so the final test stays clean.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import pickle
from dataclasses import asdict, dataclass, field

import numpy as np
import sklearn
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression, Ridge

from . import config as C
from . import models, reference
from .features import image as image_features

SENSORS = ("colour", "image")
MODEL_PATH = C.REPO / "models" / "screen_model.pkl"
SUPPLIERS_PATH = C.REPO / "user_data" / "suppliers.json"
OUTSIDE_QUANTILE = 0.999  # a sample further out than 99.9% of training samples is "outside"


@dataclass
class Reference:
    """A supplier's pure-cinnamon reference: averages over its photos and readings."""

    name: str
    lab: list  # mean L*, a*, b*
    img: list  # mean photo features (features.image.FEATURE_NAMES)
    n_photos: int
    n_readings: int
    created: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    example: bool = False


def photo_features(data: bytes):
    """Features of one uploaded photo, plus a preview with the measured area outlined.

    Mirrors how the dataset photos were processed: 512-px thumbnail, find the
    dish, measure inside 70% of its radius. Raises ValueError if no dish is found.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.draft("RGB", (1024, 1024))
            im = im.convert("RGB")
            im.thumbnail((512, 512), Image.LANCZOS)
            thumb = im.copy()
    except Exception as e:  # not an image, or unreadable
        raise ValueError(f"This file could not be read as a photo ({e.__class__.__name__}).") from e
    rgb = np.asarray(thumb, dtype=np.float64) / 255.0
    try:
        cx, cy, r = image_features.dish_circle(rgb)
        if r < 40:
            raise ValueError
        feats = image_features.features_from_thumb(thumb)
    except Exception as e:
        raise ValueError("No dish of powder could be found in this photo. Use a top-down photo of the powder in a "
                         "round dish on a light background, like the dataset photos.") from e
    if not np.isfinite(feats).all():
        raise ValueError("The powder area in this photo could not be measured.")
    preview = thumb.copy()
    draw = ImageDraw.Draw(preview)
    rr = 0.7 * r
    draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 64, 64), width=3)
    return feats.astype(float), preview


class ScreenModel:
    """The three models plus everything needed to turn raw readings into their inputs."""

    def __init__(self, blocks, band, filler, amount, X_train, margin, examples, info):
        self.col_scaler = blocks._col_scaler
        self.img_scaler, self.img_pca, self.img_scaler2 = blocks._img_scaler, blocks._img_pca, blocks._img_scaler2
        self.band, self.filler, self.amount = band, filler, amount
        self.center = X_train.mean(0)
        cov = np.cov(X_train, rowvar=False) + 1e-6 * np.eye(X_train.shape[1])
        self.icov = np.linalg.inv(cov)
        self.limit = float(np.quantile(self._distance(X_train), OUTSIDE_QUANTILE))
        self.margin = float(margin)
        self.examples = examples
        self.info = info

    def _distance(self, X):
        d = X - self.center
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.icov, d))

    def features(self, lab, img, ref: Reference) -> np.ndarray:
        """Readings relative to the supplier's reference, in the models' input space."""
        lab, img = np.atleast_2d(lab).astype(float), np.atleast_2d(img).astype(float)
        d_lab = lab - np.asarray(ref.lab, float)
        col = np.column_stack([d_lab, np.linalg.norm(d_lab, axis=1)])
        zi = self.img_scaler.transform(img - np.asarray(ref.img, float))
        i = self.img_scaler2.transform(self.img_pca.transform(zi))
        return np.hstack([self.col_scaler.transform(col), i])

    def predict(self, lab, img, ref: Reference) -> dict:
        X = self.features(lab, img, ref)
        p_band = np.zeros((len(X), len(C.BANDS)))
        p_band[:, self.band.classes_] = self.band.predict_proba(X)
        p_type = np.zeros((len(X), len(C.FOLDERS)))
        p_type[:, self.filler.classes_] = self.filler.predict_proba(X)
        dist = self._distance(X)
        return {"p_band": p_band, "p_type": p_type, "level": np.clip(self.amount.predict(X), 0, 24),
                "distance": dist, "outside": dist > self.limit}


def build(d=None, log=print, path=MODEL_PATH) -> ScreenModel:
    """Train on every open material in supplier-reference mode and save the model."""
    from . import data as data_mod

    d = d or data_mod.get()
    s = d.samples
    spectra = models.Spectra(d.X_ftir)
    is_ref = reference.split(s)
    open_rows = np.flatnonzero(~s.locked.to_numpy())
    blocks = reference.RefBlocks(d, spectra, open_rows, is_ref)
    X = blocks.X(SENSORS)
    y_band, y_type, y_level = (models.targets(s, t) for t in ("band", "type", "level"))
    blends = models.task_rows(s, "type", open_rows)
    band = LogisticRegression(class_weight="balanced", max_iter=3000).fit(X[open_rows], y_band[open_rows])
    filler = LogisticRegression(class_weight="balanced", max_iter=3000).fit(X[blends], y_type[blends])
    amount = Ridge(1.0).fit(X[open_rows], y_level[open_rows])

    margin = 5.0  # points; replaced by the fair-test figure when available
    probs_ref = C.ARTIFACTS / "band_probs_reference.npz"
    if probs_ref.exists():
        z = np.load(probs_ref)
        e3 = z["e3"][list(z["sets"]).index("colour+image")]
        if e3.shape[0] == len(s):  # the fair-test estimates belong to this dataset
            ok = np.isfinite(e3) & ~s.locked.to_numpy()
            margin = float(np.quantile(np.abs(e3[ok] - s.level.to_numpy()[ok]), 0.8))

    code = s.code.to_numpy()
    examples = {}
    for c in C.FOLDERS:
        rows = is_ref & (code == c)
        examples[c] = Reference(name=f"Example: {C.FOLDER_LABEL[c].lower()} supplier (dataset)",
                                lab=s[["L", "a", "b"]].to_numpy(float)[rows].mean(0).tolist(),
                                img=d.X_img.astype(float)[rows].mean(0).tolist(),
                                n_photos=int(rows.sum()), n_readings=int(rows.sum()), example=True)
    info = {"created": dt.datetime.now().isoformat(timespec="seconds"), "sklearn": sklearn.__version__,
            "trained_on_samples": int(len(open_rows)), "trained_on_materials": int(s.cell.iloc[open_rows].nunique()),
            "sensors": "colorimeter + photo", "mode": "supplier reference"}
    model = ScreenModel(blocks, band, filler, amount, X[open_rows], margin, examples, info)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    log(f"wrote {path} (trained on {info['trained_on_samples']} samples, {info['trained_on_materials']} materials)")
    return model


def load_model() -> ScreenModel | None:
    if not MODEL_PATH.exists():
        return None
    return pickle.loads(MODEL_PATH.read_bytes())


def load_suppliers() -> dict:
    if not SUPPLIERS_PATH.exists():
        return {}
    return {k: Reference(**v) for k, v in json.loads(SUPPLIERS_PATH.read_text()).items()}


def save_suppliers(refs: dict) -> None:
    SUPPLIERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUPPLIERS_PATH.write_text(json.dumps({k: asdict(v) for k, v in refs.items()}, indent=1))


if __name__ == "__main__":
    # Import through the package so the pickled model refers to earnedpass.deploy, not __main__.
    from earnedpass.deploy import build as _build

    _build()
