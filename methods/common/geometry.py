"""Pupil / sub-aperture geometry shared by the simulator and the pipeline.

Defines:
  - the pupil sampling grid and circular pupil mask
  - the set of valid (sufficiently illuminated) sub-apertures
  - the per-sub-aperture averaging operator used to turn a pixel-resolution
    gradient field into one (sx, sy) slope per lenslet
  - the reference (lenslet centre) coordinates on the detector

Lenslets are indexed row-major (i = row, j = column). A sub-aperture is "valid"
when its eroded illuminated fraction exceeds Config.illum_threshold; the same
valid set is used everywhere, so the interaction matrix and the measurements
always line up.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import Config


class Geometry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        n = cfg.npix
        ppl = cfg.pix_per_lenslet

        # Pupil-plane coordinates, centred, normalised to the unit disk.
        ax = (np.arange(n) + 0.5 - n / 2.0)
        xx, yy = np.meshgrid(ax, ax)
        radius_pix = n / 2.0
        self.rho = np.sqrt(xx ** 2 + yy ** 2) / radius_pix
        self.theta = np.arctan2(yy, xx)
        self.pupil_mask = self.rho <= 1.0

        # Erode by one pixel: finite-difference gradients at the pupil edge use
        # an outside neighbour and are unreliable, so we exclude that ring from
        # every sub-aperture average (consistently in forward and inverse).
        self.grad_mask = ndimage.binary_erosion(self.pupil_mask)

        # Reshape helper: (n, n) -> (n_lenslets, ppl, n_lenslets, ppl)
        self._ppl = ppl
        self._nl = cfg.n_lenslets

        # Per-lenslet illuminated pixel count (using the eroded mask).
        gm = self.grad_mask.astype(float)
        counts = gm.reshape(self._nl, ppl, self._nl, ppl).sum(axis=(1, 3))
        frac = counts / (ppl * ppl)
        self.valid = frac >= cfg.illum_threshold  # (n_lenslets, n_lenslets) bool
        self.valid_idx = np.argwhere(self.valid)  # rows of [i, j]
        self.n_valid = int(self.valid.sum())

        # Detector-pixel coordinates of each lenslet centre (the reference grid
        # for a flat wavefront).
        ji = np.arange(self._nl)
        centres = (ji + 0.5) * ppl
        self.ref_x = np.array([centres[j] for (i, j) in self.valid_idx])
        self.ref_y = np.array([centres[i] for (i, j) in self.valid_idx])

    def average_per_subaperture(self, field: np.ndarray) -> np.ndarray:
        """Weighted mean of `field` over each valid sub-aperture (eroded mask).

        Returns a 1-D array of length n_valid, ordered as `valid_idx`.
        """
        ppl = self._ppl
        nl = self._nl
        w = self.grad_mask.astype(float)
        num = (field * w).reshape(nl, ppl, nl, ppl).sum(axis=(1, 3))
        den = w.reshape(nl, ppl, nl, ppl).sum(axis=(1, 3))
        den = np.where(den > 0, den, 1.0)
        avg = num / den
        return avg[self.valid]
