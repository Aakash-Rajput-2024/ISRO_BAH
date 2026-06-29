"""Tier 1A -- reconstruction fidelity.

Characterises the modal reconstructor beyond the simple round-trip: linear-
algebra sanity (R is a left inverse of D), real per-mode cross-talk through the
full forward+inverse chain (including centroiding), amplitude linearity / the
dynamic-range limit, and how residual error responds to noise and to the number
of reconstructed modes.
"""
import numpy as np
import pytest

from methods.modal_zernike import Config, WFSPipeline
from methods.modal_zernike.reconstruct import ModalReconstructor
from methods.common.simulate import flat_frame, render_frame
from validations import metrics as m


def _transfer_matrix(pipe, amp=0.2):
    """Recovered-coefficient matrix M: column k = recover(inject unit in mode k).

    Built noiselessly so it isolates discretisation + centroiding cross-talk
    from photon/read noise. M[j,k] is the leakage of injected mode k into
    recovered mode j; ideally the identity.
    """
    nm = pipe.cfg.n_modes
    M = np.zeros((nm, nm))
    for k in range(nm):
        c = np.zeros(nm)
        c[k] = amp
        phase = pipe.recon.wavefront_from_coeffs(c)
        M[:, k] = pipe.process(render_frame(phase, pipe.cfg, pipe.geom)).coeffs / amp
    return M


@pytest.mark.fast
def test_reconstructor_is_left_inverse(pipe, tol):
    """R @ D must equal the identity (D has full column rank)."""
    RD = m.crosstalk_matrix(pipe.recon.R, pipe.recon.D)
    np.testing.assert_allclose(RD, np.eye(RD.shape[0]), atol=tol["left_inverse"])


@pytest.mark.fast
def test_empirical_mode_transfer_is_diagonal(pipe, tol):
    """Through the full pipeline, each injected mode is recovered with near-unit
    gain and small leakage into the other modes."""
    M = _transfer_matrix(pipe)
    diag = np.diag(M)
    offdiag = M - np.diag(diag)
    assert diag.min() >= tol["crosstalk_diag_lo"]
    assert diag.max() <= tol["crosstalk_diag_hi"]
    assert np.abs(offdiag).max() < tol["crosstalk_offdiag"]


@pytest.mark.fast
def test_single_mode_linearity(pipe):
    """Recovered amplitude is linear in injected amplitude within the linear
    (small-aberration) regime -- the precondition for matrix reconstruction."""
    j = 5  # astigmatism
    amps = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    rec = []
    for a in amps:
        c = np.zeros(pipe.cfg.n_modes)
        c[j - 2] = a
        phase = pipe.recon.wavefront_from_coeffs(c)
        rec.append(pipe.process(render_frame(phase, pipe.cfg, pipe.geom)).coeffs[j - 2])
    rec = np.array(rec)
    # Linear fit; correlation must be essentially perfect and gain near unity.
    slope, intercept = np.polyfit(amps, rec, 1)
    resid = rec - (slope * amps + intercept)
    r2 = 1.0 - resid.var() / rec.var()
    assert r2 > 0.999
    assert 0.9 < slope < 1.05


@pytest.mark.slow
def test_residual_decreases_with_snr(pipe, rng):
    """Reconstruction residual RMS shrinks as detector SNR improves."""
    cfg = pipe.cfg
    mask = pipe.geom.pupil_mask
    c = np.zeros(cfg.n_modes)
    c[5 - 2], c[8 - 2] = 0.6, -0.4
    truth = pipe.recon.wavefront_from_coeffs(c)

    def mean_residual(peak, n=12):
        out = []
        for _ in range(n):
            frame = render_frame(truth, cfg, pipe.geom, peak=peak,
                                 read_noise=5.0, photon_noise=True, rng=rng)
            recon = pipe.process(frame).wavefront
            out.append(m.residual_rms(truth, recon, mask))
        return np.mean(out)

    hi_snr = mean_residual(peak=5000.0)
    lo_snr = mean_residual(peak=200.0)
    assert hi_snr < lo_snr


@pytest.mark.slow
def test_modal_fitting_error_decreases_with_more_modes(rng):
    """Adding Zernike modes reduces the reconstruction residual of a turbulent
    wavefront (modal fitting error, tracking Noll's residual-variance tail)."""
    from methods.common.phasescreen import kolmogorov_screen

    cfg_lo = Config(n_modes=6)
    cfg_hi = Config(n_modes=20)
    pipes = []
    for cfg in (cfg_lo, cfg_hi):
        p = WFSPipeline(cfg)
        p.calibrate(flat_frame(cfg, p.geom))
        pipes.append(p)

    base = Config()
    mask = pipes[0].geom.pupil_mask
    res_lo, res_hi = [], []
    for _ in range(10):
        ph = kolmogorov_screen(base.npix, base.pupil_dx, base.pupil_diameter / 6.0, rng)
        ph = (ph - ph[mask].mean()) * mask
        frame_lo = render_frame(ph, cfg_lo, pipes[0].geom, rng=rng)
        frame_hi = render_frame(ph, cfg_hi, pipes[1].geom, rng=rng)
        res_lo.append(m.residual_rms(ph, pipes[0].process(frame_lo).wavefront, mask))
        res_hi.append(m.residual_rms(ph, pipes[1].process(frame_hi).wavefront, mask))
    assert np.mean(res_hi) < np.mean(res_lo)
