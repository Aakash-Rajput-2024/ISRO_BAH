"""Input preprocessing for the ResU-Net (Method 3).

Two pieces from Noel et al. (2023):
  - intensity normalisation of each frame to [0, 1];
  - the "average-added" representation (Fig. 4/5): superimpose the time-averaged
    spot pattern onto each instantaneous frame so every sub-aperture encodes its
    deviation from the reference position as a single blended feature -- letting
    the network learn slopes without an explicit reference subtraction.
"""
from __future__ import annotations

import numpy as np


def normalize(frame: np.ndarray) -> np.ndarray:
    """Scale a frame to [0, 1] (guarding against an all-zero frame)."""
    frame = np.asarray(frame, dtype=np.float32)
    peak = float(frame.max())
    return frame / peak if peak > 0 else frame


def average_added(frame: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
    """Normalised frame, optionally with the (normalised) average frame added.

    `ref` is the time-averaged frame over the dataset; pass None to disable.
    The sum is renormalised back to [0, 1].
    """
    x = normalize(frame)
    if ref is not None:
        x = x + normalize(ref)
        peak = float(x.max())
        if peak > 0:
            x = x / peak
    return x.astype(np.float32)
