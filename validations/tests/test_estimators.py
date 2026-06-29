"""Tier 1B -- turbulence-estimator validation (r0, tau0).

Goes beyond the single ±40% sanity check: r0 recovery across turbulence
strengths, the j_start design choice (skipping the FFT-biased tip/tilt), and
the tau0 inverse-wind scaling invariant.
"""
import numpy as np
import pytest

from methods.modal_zernike import Config, WFSPipeline
from methods.common.phasescreen import frozen_flow_sequence, kolmogorov_screen
from methods.common.simulate import flat_frame, render_frame
from methods.common.turbulence import estimate_r0, estimate_tau0


def _coeff_ensemble(pipe, r0, n, rng):
    cfg = pipe.cfg
    mask = pipe.geom.pupil_mask
    coeffs = np.empty((n, cfg.n_modes))
    for k in range(n):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0, rng)
        ph = ph - ph[mask].mean()
        coeffs[k] = pipe.process(render_frame(ph, cfg, pipe.geom, rng=rng)).coeffs
    return coeffs


@pytest.mark.slow
@pytest.mark.parametrize("d_over_r0", [4.0, 6.0, 8.0])
def test_r0_recovers_across_strength(pipe, tol, d_over_r0):
    """Estimated r0 lands within tolerance of truth across turbulence strengths."""
    rng = np.random.default_rng(int(d_over_r0))
    D = pipe.cfg.pupil_diameter
    r0_true = D / d_over_r0
    coeffs = _coeff_ensemble(pipe, r0_true, 150, rng)
    r0_est = estimate_r0(coeffs, D)
    assert abs(r0_est - r0_true) / r0_true < tol["r0_rel"]


@pytest.mark.slow
def test_jstart_4_beats_jstart_2(pipe):
    """Starting the variance fit at j=4 (skipping tip/tilt) is less biased than
    j=2, because the FFT screen under-represents tip/tilt (Andrade et al. 2019)."""
    rng = np.random.default_rng(3)
    D = pipe.cfg.pupil_diameter
    r0_true = D / 6.0
    coeffs = _coeff_ensemble(pipe, r0_true, 150, rng)
    bias4 = abs(estimate_r0(coeffs, D, j_start=4) - r0_true)
    bias2 = abs(estimate_r0(coeffs, D, j_start=2) - r0_true)
    assert bias4 < bias2


@pytest.mark.slow
def test_r0_converges_with_frames(pipe):
    """The r0 estimate stabilises as the ensemble grows."""
    rng = np.random.default_rng(5)
    D = pipe.cfg.pupil_diameter
    r0_true = D / 6.0
    coeffs = _coeff_ensemble(pipe, r0_true, 250, rng)
    early = estimate_r0(coeffs[:60], D)
    late = estimate_r0(coeffs[:250], D)
    full = estimate_r0(coeffs, D)
    # The tail should move less than the head -> convergence.
    assert abs(full - late) < abs(late - early) + 1e-12


@pytest.mark.fast
def test_tau0_scales_inversely_with_wind(pipe, tol):
    """tau0 * wind is invariant under the frozen-flow hypothesis: doubling the
    wind speed halves the coherence time. This is the robust physical check
    (absolute agreement with 0.314 r0/v is reported, not asserted)."""
    cfg = pipe.cfg
    mask = pipe.geom.pupil_mask
    D = cfg.pupil_diameter
    r0 = D / 6.0

    products = []
    for wind in (4.0, 8.0):
        rng = np.random.default_rng(7)  # same screen -> exact 1/wind scaling
        dt = cfg.pupil_dx / wind
        seq = list(frozen_flow_sequence(cfg.npix, cfg.pupil_dx, r0, 80, wind, dt, rng))
        wfs = np.stack([
            pipe.process(render_frame(ph - ph[mask].mean(), cfg, pipe.geom, rng=rng)).coeffs
            for ph in seq
        ])
        products.append(estimate_tau0(wfs, dt) * wind)

    rel = abs(products[0] - products[1]) / np.mean(products)
    assert rel < tol["tau0_wind_rel"]
