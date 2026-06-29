"""Inference wrapper for the trained ResU-Net (Method 3).

Exposes the same comparable interface as the modal method -- `process(frame)`
returning a `.wavefront` (pupil phase map [rad]) and `.coeffs` (Zernike) -- so
Method 3 drops straight into the same plots/benchmark as Method 1.

The network is zonal: it predicts a coarse wavefront map. We upsample it to the
pupil grid, remove piston, and project onto the Zernike basis to obtain the
coefficients (and hence r0) for comparison with the modal method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..common.config import Config
from ..common.geometry import Geometry
from ..common.zernike import zernike_basis
from .model import ResUNet, pick_device
from .preprocess import average_added


@dataclass
class DeepResult:
    wavefront: np.ndarray   # pupil phase map [rad]
    coeffs: np.ndarray      # Zernike coefficients [rad], Noll j = 2..n_modes+1


class ResUNetReconstructor:
    def __init__(self, checkpoint_path: str, cfg: Config | None = None,
                 device: torch.device | None = None):
        self.cfg = cfg or Config()
        self.geom = Geometry(self.cfg)
        self.device = device or pick_device()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.target = ckpt["target"]
        self.wf_scale = ckpt["wf_scale"]
        self.npix = ckpt["npix"]
        self.avg_added = ckpt["avg_added"]
        self.mean_frame = ckpt["mean_frame"]
        self.model = ResUNet(target=self.target, base=ckpt["base"]).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        # Zernike basis + projection operator for coeff extraction.
        modes = list(range(2, self.cfg.n_modes + 2))
        self.zmaps = zernike_basis(modes, self.geom.rho, self.geom.theta,
                                   self.geom.pupil_mask)
        self._A = self.zmaps[:, self.geom.pupil_mask].T  # (n_pix, n_modes)

    def predict_wavefront(self, frame: np.ndarray) -> np.ndarray:
        ref = self.mean_frame if self.avg_added else None
        x = average_added(frame, ref)
        xt = torch.from_numpy(x)[None, None].float().to(self.device)
        with torch.no_grad():
            coarse = self.model(xt).cpu().numpy()[0, 0]  # (target, target), [-1,1]
        coarse = coarse * self.wf_scale
        # Upsample to the pupil grid and mask.
        ct = torch.from_numpy(coarse)[None, None].float()
        full = F.interpolate(ct, size=(self.cfg.npix, self.cfg.npix),
                             mode="bilinear", align_corners=False).numpy()[0, 0]
        full = full * self.geom.pupil_mask
        # Remove piston over the pupil (the modal basis excludes it).
        m = self.geom.pupil_mask
        full[m] -= full[m].mean()
        return full * m

    def process(self, frame: np.ndarray) -> DeepResult:
        wf = self.predict_wavefront(frame)
        coeffs, *_ = np.linalg.lstsq(self._A, wf[self.geom.pupil_mask], rcond=None)
        return DeepResult(wavefront=wf, coeffs=coeffs)
