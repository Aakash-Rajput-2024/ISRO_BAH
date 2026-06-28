"""End-to-end pipeline: detector frame -> slopes -> Zernike coeffs -> wavefront.

Usage:
    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_image)          # set reference spot positions
    out = pipe.process(frame)           # dict: slopes, coeffs, wavefront
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centroid import centroid_frame
from .config import Config
from .geometry import Geometry
from .reconstruct import ModalReconstructor


@dataclass
class FrameResult:
    cx: np.ndarray            # absolute spot centroids x [px]
    cy: np.ndarray            # absolute spot centroids y [px]
    slopes: np.ndarray        # concat(sx, sy) [rad/m]
    coeffs: np.ndarray        # Zernike coefficients [rad], Noll j = 2..n_modes+1
    wavefront: np.ndarray     # reconstructed pupil phase map [rad]


class WFSPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.geom = Geometry(self.cfg)
        self.recon = ModalReconstructor(self.cfg, self.geom)
        # Default reference = geometric lenslet centres; overwritten by calibrate.
        self.ref_x = self.geom.ref_x.copy()
        self.ref_y = self.geom.ref_y.copy()

    def calibrate(self, flat_image: np.ndarray) -> None:
        """Measure reference spot positions from a flat-wavefront frame."""
        self.ref_x, self.ref_y = centroid_frame(flat_image, self.cfg, self.geom)

    def slopes_from_frame(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cx, cy = centroid_frame(image, self.cfg, self.geom)
        dx_px = cx - self.ref_x
        dy_px = cy - self.ref_y
        sx = dx_px * self.cfg.slope_scale
        sy = dy_px * self.cfg.slope_scale
        return cx, cy, np.concatenate([sx, sy])

    def process(self, image: np.ndarray) -> FrameResult:
        cx, cy, slopes = self.slopes_from_frame(image)
        coeffs = self.recon.coeffs_from_slopes(slopes)
        wavefront = self.recon.wavefront_from_coeffs(coeffs)
        return FrameResult(cx=cx, cy=cy, slopes=slopes, coeffs=coeffs, wavefront=wavefront)
