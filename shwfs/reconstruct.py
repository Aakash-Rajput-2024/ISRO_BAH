"""Modal (Zernike) wavefront reconstruction.

We measure one slope pair (sx, sy) per valid sub-aperture. The wavefront is
written as W = sum_k a_k Z_k, so the slopes are linear in the coefficients:

    s = D a            D = modal interaction matrix  (2 * n_valid, n_modes)
    a = pinv(D) s      least-squares reconstruction

D is built once from the (known) geometry; per frame the reconstruction is a
single matrix-vector product `R @ s` -- this is the step that must run in a few
milliseconds in the eventual C port.

Slopes here are phase gradients in rad/m (consistent with the Zernike maps,
which carry coefficients in radians of phase).
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .geometry import Geometry
from .zernike import zernike_basis


class ModalReconstructor:
    def __init__(self, cfg: Config, geom: Geometry):
        self.cfg = cfg
        self.geom = geom
        self.modes = list(range(2, cfg.n_modes + 2))  # Noll j, skip piston (j=1)

        # Zernike phase maps over the pupil (radians per unit coefficient).
        self.zmaps = zernike_basis(self.modes, geom.rho, geom.theta, geom.pupil_mask)

        self.D = self._build_interaction_matrix()
        self.R = np.linalg.pinv(self.D)

    def _build_interaction_matrix(self) -> np.ndarray:
        dx = self.cfg.pupil_dx
        cols = []
        for zmap in self.zmaps:
            gy, gx = np.gradient(zmap, dx)  # gy = d/drow(y), gx = d/dcol(x)
            sx = self.geom.average_per_subaperture(gx)
            sy = self.geom.average_per_subaperture(gy)
            cols.append(np.concatenate([sx, sy]))
        return np.stack(cols, axis=1)  # (2 * n_valid, n_modes)

    def coeffs_from_slopes(self, slopes: np.ndarray) -> np.ndarray:
        """slopes = concat(sx, sy) over valid sub-apertures -> Zernike coeffs."""
        return self.R @ slopes

    def wavefront_from_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct the pupil phase map [rad] from Zernike coefficients."""
        return np.tensordot(coeffs, self.zmaps, axes=(0, 0))
