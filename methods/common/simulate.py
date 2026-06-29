"""Synthetic SH-WFS frame generator (the test oracle).

Given a pupil phase map [rad], compute the true per-sub-aperture phase gradient
(same averaging operator the reconstructor uses), convert it to a spot shift on
the detector, and render one Gaussian spot per valid lenslet. Optional photon +
read noise. Because forward and inverse share the exact averaging operator, a
noiseless frame round-trips back to the input Zernike coefficients -- that is
what the recovery test checks.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .geometry import Geometry


def true_slopes(phase: np.ndarray, cfg: Config, geom: Geometry) -> np.ndarray:
    """Per-sub-aperture phase gradient [rad/m]; returns concat(sx, sy)."""
    gy, gx = np.gradient(phase, cfg.pupil_dx)
    sx = geom.average_per_subaperture(gx)
    sy = geom.average_per_subaperture(gy)
    return np.concatenate([sx, sy])


def render_frame(
    phase: np.ndarray,
    cfg: Config,
    geom: Geometry,
    spot_sigma: float = 1.5,
    peak: float = 1000.0,
    read_noise: float = 0.0,
    photon_noise: bool = False,
    rng: np.random.Generator | None = None,
    bias: float = 0.0,
    dark: float = 0.0,
    full_well: float | None = None,
    quantize_bits: int | None = None,
) -> np.ndarray:
    """Render a detector frame for the given pupil phase map.

    Detector-realism options (all default to no-op, so existing callers are
    unchanged): `dark` (added dark signal before photon noise), `bias` (added
    offset after noise), `full_well` (saturation clip), `quantize_bits` (digitise
    to N-bit integers, e.g. 8 for a .bmp). With these set the frame resembles a
    real science-camera readout rather than an idealised float image.
    """
    if rng is None:
        rng = np.random.default_rng()
    ppl = cfg.pix_per_lenslet
    n = cfg.npix
    img = np.zeros((n, n), dtype=float)

    s = true_slopes(phase, cfg, geom)
    nv = geom.n_valid
    sx, sy = s[:nv], s[nv:]
    shift_x = sx / cfg.slope_scale  # phase gradient -> spot shift [px]
    shift_y = sy / cfg.slope_scale

    loc = np.arange(ppl) + 0.5
    lx, ly = np.meshgrid(loc, loc)
    for k, (i, j) in enumerate(geom.valid_idx):
        y0, x0 = i * ppl, j * ppl
        cx = ppl / 2.0 + shift_x[k]
        cy = ppl / 2.0 + shift_y[k]
        spot = peak * np.exp(
            -((lx - cx) ** 2 + (ly - cy) ** 2) / (2.0 * spot_sigma ** 2)
        )
        img[y0 : y0 + ppl, x0 : x0 + ppl] += spot

    if dark > 0:
        img = img + dark
    if photon_noise:
        img = rng.poisson(np.clip(img, 0, None)).astype(float)
    if read_noise > 0:
        img = img + rng.normal(0.0, read_noise, img.shape)
    if bias > 0:
        img = img + bias
    if full_well is not None:
        img = np.clip(img, 0.0, full_well)
    if quantize_bits is not None:
        levels = 2 ** quantize_bits - 1
        hi = full_well if full_well is not None else float(img.max() or 1.0)
        img = np.round(np.clip(img / hi, 0.0, 1.0) * levels)  # digital counts
    return img


def flat_frame(cfg: Config, geom: Geometry, **kw) -> np.ndarray:
    """Calibration frame for a flat wavefront (spots at lenslet centres)."""
    return render_frame(np.zeros((cfg.npix, cfg.npix)), cfg, geom, **kw)
