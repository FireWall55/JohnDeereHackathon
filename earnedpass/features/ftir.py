"""FTIR features. Units are assumed to be transmittance (values sit near 1.0)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def absorbance(X: np.ndarray) -> np.ndarray:
    return -np.log10(np.clip(X, 1e-6, None)).astype(np.float32)


def first_derivative(A: np.ndarray) -> np.ndarray:
    """Savitzky-Golay first derivative, window 15, order 2. Removes flat offsets."""
    return savgol_filter(A, window_length=15, polyorder=2, deriv=1, axis=1).astype(np.float32)
