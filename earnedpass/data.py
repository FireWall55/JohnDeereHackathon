"""Build and cache the data contract; load it for modelling and the app."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import groups, load
from .features import image


@dataclass
class Data:
    samples: pd.DataFrame
    X_ftir: np.ndarray  # raw instrument values, 1,950 x 2,523
    wavenumbers: np.ndarray
    X_img: np.ndarray

    @property
    def dev(self) -> np.ndarray:
        return np.flatnonzero(~self.samples.locked.to_numpy())

    @property
    def locked(self) -> np.ndarray:
        return np.flatnonzero(self.samples.locked.to_numpy())


def build(force=False, log=print) -> None:
    A = C.ARTIFACTS
    A.mkdir(parents=True, exist_ok=True)
    if force or not (A / "samples.parquet").exists():
        log("reading colour and FTIR spreadsheets (about a minute)...")
        s, X, wn, audit = load.build_samples()
        log("reading photo timestamps and building groups...")
        s = groups.add_groups(s)
        np.save(A / "X_ftir.npy", X)
        np.save(A / "wavenumbers.npy", wn)
        s.to_parquet(A / "samples.parquet", index=False)
        (A / "load_audit.json").write_text(json.dumps(audit, indent=1))
    if force or not (A / "X_img.npy").exists():
        log("making thumbnails and image features (a few minutes)...")
        s = pd.read_parquet(A / "samples.parquet")
        np.save(A / "X_img.npy", image.build(s, C.THUMBS))


def get() -> Data:
    A = C.ARTIFACTS
    return Data(
        pd.read_parquet(A / "samples.parquet"),
        np.load(A / "X_ftir.npy"),
        np.load(A / "wavenumbers.npy"),
        np.load(A / "X_img.npy"),
    )


if __name__ == "__main__":
    build()
