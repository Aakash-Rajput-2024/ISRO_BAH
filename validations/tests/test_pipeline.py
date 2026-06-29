"""Ground-truth round-trip tests for the SH-WFS pipeline (Tier 1, fast).

These are the original oracle checks the eventual C port must reproduce. They
use the shared `pipe`/`cfg`/`tol` fixtures from conftest.py.
"""
import numpy as np
import pytest

from methods.modal_zernike import Config, WFSPipeline
from methods.common.phasescreen import kolmogorov_screen
from methods.common.simulate import flat_frame, render_frame
from methods.common.turbulence import estimate_r0


@pytest.mark.fast
def test_zernike_recovery(pipe, tol):
    """A known Zernike aberration must round-trip through the pipeline.

    Inject astigmatism (j=5), coma (j=8) and spherical (j=11); the noiseless
    forward model shares the reconstructor's averaging operator, so recovery is
    limited only by sub-pixel centroiding.
    """
    injected = {5: 0.8, 8: -0.5, 11: 0.3}  # Noll j -> coefficient [rad]
    coeffs_in = np.zeros(pipe.cfg.n_modes)
    for j, a in injected.items():
        coeffs_in[j - 2] = a

    phase = pipe.recon.wavefront_from_coeffs(coeffs_in)
    frame = render_frame(phase, pipe.cfg, pipe.geom)
    out = pipe.process(frame)

    np.testing.assert_allclose(out.coeffs, coeffs_in, atol=tol["single_mode_recovery_rad"])


@pytest.mark.fast
def test_flat_wavefront_is_zero(pipe):
    """A flat wavefront produces ~zero slopes and coefficients."""
    frame = flat_frame(pipe.cfg, pipe.geom)
    out = pipe.process(frame)
    assert np.abs(out.slopes).max() < 1e-6 * pipe.cfg.slope_scale + 1e3
    assert np.abs(out.coeffs).max() < 0.02


@pytest.mark.slow
def test_r0_estimate_recovers_truth(tol):
    """Estimated r0 from an ensemble should land within tolerance of truth."""
    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, pipe.geom))
    rng = np.random.default_rng(0)
    D = cfg.pupil_diameter
    r0_true = D / 6.0

    n = 150
    coeffs = np.empty((n, cfg.n_modes))
    mask = pipe.geom.pupil_mask
    for k in range(n):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0_true, rng)
        ph = ph - ph[mask].mean()
        coeffs[k] = pipe.process(render_frame(ph, cfg, pipe.geom, rng=rng)).coeffs

    r0_est = estimate_r0(coeffs, D)
    assert (1 - tol["r0_rel"]) * r0_true < r0_est < (1 + tol["r0_rel"]) * r0_true
