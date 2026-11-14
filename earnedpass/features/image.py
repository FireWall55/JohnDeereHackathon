"""One pass over the photos: 512-px thumbnail, crop inside the dish, features."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

HUE_BINS = np.linspace(40.0, 90.0, 13)
FEATURE_NAMES = (
    ["img_L_mean", "img_a_mean", "img_b_mean", "img_L_sd", "img_a_sd", "img_b_sd"]
    + [f"img_hue_{int(lo)}" for lo in HUE_BINS[:-1]]
    + ["img_laplacian_var"]
)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] to CIE L*a*b* (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def dish_circle(rgb: np.ndarray):
    """Centre and radius of the powder, found from colour saturation."""
    sat = rgb.max(-1) - rgb.min(-1)
    mask = sat > 0.18
    cols, rows = mask.sum(0), mask.sum(1)
    xs = np.flatnonzero(cols > 0.25 * cols.max())
    ys = np.flatnonzero(rows > 0.25 * rows.max())
    cx, cy = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
    r = min(xs[-1] - xs[0], ys[-1] - ys[0]) / 2
    return cx, cy, r


def features_from_thumb(im: Image.Image) -> np.ndarray:
    rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    cx, cy, r = dish_circle(rgb)
    yy, xx = np.mgrid[: rgb.shape[0], : rgb.shape[1]]
    inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.7 * r) ** 2
    lab = rgb_to_lab(rgb)
    px = lab[inner]
    hue = np.degrees(np.arctan2(px[:, 2], px[:, 1]))
    hist = np.histogram(np.clip(hue, HUE_BINS[0], HUE_BINS[-1] - 1e-9), bins=HUE_BINS)[0] / len(px)
    h = int(0.7 * r / np.sqrt(2))
    sq = lab[int(cy) - h : int(cy) + h, int(cx) - h : int(cx) + h, 0]
    return np.concatenate([px.mean(0), px.std(0), hist, [ndimage.laplace(sq).var()]])


def _process(job):
    src, dst = job
    with Image.open(src) as im:
        im.draft("RGB", (1024, 1024))
        im = im.convert("RGB")
        im.thumbnail((512, 512), Image.LANCZOS)
    im.save(dst, quality=90)
    return features_from_thumb(im)


def build(samples, thumbs_dir, workers=None) -> np.ndarray:
    os.makedirs(thumbs_dir, exist_ok=True)
    jobs = [(p, os.path.join(thumbs_dir, f"{sid}.jpg")) for sid, p in zip(samples.sample_id, samples.img_path)]
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        feats = list(ex.map(_process, jobs, chunksize=8))
    return np.vstack(feats).astype(np.float32)
