"""Spot centroiding.

For each valid sub-aperture we take the fixed lenslet cell (pix_per_lenslet
square) as the search window and compute a thresholded centre-of-gravity (CoG):

    x_c = sum(x * I') / sum(I'),   I' = max(I - threshold, 0)

Thresholding removes background / read noise that would otherwise pull the CoG
toward the window centre. This is the simplest fast estimator; weighted CoG,
correlation, or Gaussian-fit centroiding can be dropped in later behind the same
interface.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .geometry import Geometry


def centroid_frame(
    image: np.ndarray,
    cfg: Config,
    geom: Geometry,
    threshold_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return absolute detector-pixel centroids (cx, cy) for valid sub-apertures.

    Ordered to match geom.valid_idx. If a window has no signal above threshold
    the centroid falls back to the window centre.
    """
    ppl = cfg.pix_per_lenslet
    cx = np.empty(geom.n_valid)
    cy = np.empty(geom.n_valid)

    # Local pixel coordinates within a window (centre of each pixel).
    loc = np.arange(ppl) + 0.5
    lx, ly = np.meshgrid(loc, loc)

    for k, (i, j) in enumerate(geom.valid_idx):
        y0, x0 = i * ppl, j * ppl
        win = image[y0 : y0 + ppl, x0 : x0 + ppl]
        thr = threshold_frac * win.max()
        w = np.clip(win - thr, 0.0, None)
        total = w.sum()
        if total <= 0:
            cx[k] = x0 + ppl / 2.0
            cy[k] = y0 + ppl / 2.0
            continue
        cx[k] = x0 + (w * lx).sum() / total
        cy[k] = y0 + (w * ly).sum() / total
    return cx, cy
