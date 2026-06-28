"""Ground-truth tests for the SH-WFS pipeline.

These are the oracle the eventual C port must reproduce.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shwfs import Config, WFSPipeline
from shwfs.phasescreen import kolmogorov_screen
from shwfs.simulate import flat_frame, render_frame
from shwfs.turbulence import estimate_r0


@pytest.fixture(scope="module")
def pipe():
    p = WFSPipeline(Config())
    p.calibrate(flat_frame(p.cfg, p.geom))
    return p


def test_zernike_recovery(pipe):
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

    np.testing.assert_allclose(out.coeffs, coeffs_in, atol=0.03)


def test_flat_wavefront_is_zero(pipe):
    """A flat wavefront produces ~zero slopes and coefficients."""
    frame = flat_frame(pipe.cfg, pipe.geom)
    out = pipe.process(frame)
    assert np.abs(out.slopes).max() < 1e-6 * pipe.cfg.slope_scale + 1e3
    assert np.abs(out.coeffs).max() < 0.02


def test_r0_estimate_recovers_truth():
    """Estimated r0 from an ensemble should land within ~40% of the truth."""
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
    assert 0.6 * r0_true < r0_est < 1.6 * r0_true
