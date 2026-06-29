"""Tier 1C -- phase-screen statistics.

Validates the Kolmogorov screen generator *itself* (independently of the
reconstruction round-trip) against turbulence theory. The plain FFT method is
known to under-represent the lowest spatial frequencies, biasing absolute
magnitudes low (Lane et al. 1992); we therefore validate the power-law SHAPE
(slopes) and document the magnitude deficit rather than asserting on it.
"""
import numpy as np
import pytest

from methods.modal_zernike import Config
from methods.common.phasescreen import kolmogorov_screen
from methods.common.simulate import flat_frame, render_frame
from validations import metrics as m


def _ensemble_structure_fn(cfg, r0, n, rng, max_lag):
    mask = (np.add.outer(*[(np.arange(cfg.npix) - cfg.npix / 2.0) ** 2] * 2)
            <= (cfg.npix / 2.0) ** 2)
    acc = None
    for _ in range(n):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0, rng)
        ph = ph - ph[mask].mean()
        r, d = m.structure_function(ph, cfg.pupil_dx, max_lag=max_lag)
        acc = d if acc is None else acc + d
    return r, acc / n


@pytest.mark.fast
def test_structure_function_slope(cfg, tol):
    """Mid-range structure function follows the Kolmogorov r^(5/3) power law."""
    rng = np.random.default_rng(0)
    r0 = cfg.pupil_diameter / 6.0
    r, d = _ensemble_structure_fn(cfg, r0, 6, rng, max_lag=cfg.npix // 2)
    # Small-to-mid range only: skip the smallest lags (pixelisation) and the
    # larger lags where the FFT low-frequency deficit flattens the curve below
    # the r^(5/3) law (Lane et al. 1992).
    lo, hi = 2, cfg.npix // 16
    slope = m.loglog_slope(r[lo:hi], d[lo:hi])
    assert abs(slope - tol["struct_fn_slope"]) < tol["struct_fn_slope_tol"]


@pytest.mark.fast
def test_radial_psd_slope(cfg, tol):
    """Mid-frequency radial PSD follows the Kolmogorov f^(-11/3) power law."""
    rng = np.random.default_rng(1)
    r0 = cfg.pupil_diameter / 6.0
    ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0, rng)
    f, psd = m.radial_psd(ph, cfg.pupil_dx)
    lo, hi = 3, cfg.npix // 4
    slope = m.loglog_slope(f[lo:hi], psd[lo:hi])
    assert abs(slope - tol["psd_slope"]) < tol["psd_slope_tol"]


@pytest.mark.slow
def test_zernike_variance_spectrum_vs_noll(cfg, tol):
    """Per-mode coefficient variance vs Noll (1976).

    Tip/tilt (j=2,3) are strongly suppressed by the FFT low-frequency deficit;
    mid-order modes (j>=4) recover most of the predicted variance. This is the
    quantitative reason estimate_r0 starts the fit at j=4.
    """
    pipe_cfg = cfg
    from methods.modal_zernike import WFSPipeline
    pipe = WFSPipeline(pipe_cfg)
    pipe.calibrate(flat_frame(pipe_cfg, pipe.geom))
    mask = pipe.geom.pupil_mask
    D = pipe_cfg.pupil_diameter
    r0 = D / 6.0

    rng = np.random.default_rng(0)
    N = 200
    coeffs = np.empty((N, pipe_cfg.n_modes))
    for k in range(N):
        ph = kolmogorov_screen(pipe_cfg.npix, pipe_cfg.pupil_dx, r0, rng)
        ph = ph - ph[mask].mean()
        coeffs[k] = pipe.process(render_frame(ph, pipe_cfg, pipe.geom, rng=rng)).coeffs

    meas = m.zernike_variance_spectrum(coeffs)
    js = list(range(2, pipe_cfg.n_modes + 2))
    theory = m.noll_mode_variance_spectrum(js, D, r0)

    # Tip/tilt deficit (j=2,3 -> columns 0,1).
    assert (meas[:2] / theory[:2]).max() < tol["tiptilt_deficit_ratio"]
    # Mid-order captured fraction sits in the documented band.
    mid_ratio = meas[2:].sum() / theory[2:].sum()
    assert tol["midorder_ratio_lo"] < mid_ratio < tol["midorder_ratio_hi"]
