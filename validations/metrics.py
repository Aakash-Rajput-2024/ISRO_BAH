"""Reusable, plot-free numeric kernels for validation.

Everything here returns plain numbers / arrays so the same code backs both the
pytest assertions and the visual report. Keep this module dependency-light
(numpy only) and free of matplotlib.

References:
  - Kolmogorov structure function: D_phi(r) = 6.88 (r/r0)^(5/3)  (Fried 1965;
    see Hardy 1998, eq. 3.30).
  - Per-mode Zernike-Kolmogorov residual variance: Noll (1976), Table IV.
"""
from __future__ import annotations

import numpy as np

from methods.common.turbulence import _noll_delta


# --------------------------------------------------------------------------- #
# Reconstruction-fidelity metrics
# --------------------------------------------------------------------------- #
def crosstalk_matrix(R: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Mode-to-mode transfer of the reconstructor: ideally the identity.

    R @ D maps injected Zernike coefficients to recovered ones; the diagonal is
    the per-mode gain and off-diagonals are inter-mode cross-talk.
    """
    return R @ D


def residual_rms(truth: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> float:
    """RMS of (truth - recon) over the pupil mask [same units as the maps]."""
    res = (truth - recon)[mask]
    return float(np.sqrt(np.mean(res ** 2)))


def snr_of_frame(frame: np.ndarray) -> float:
    """Empirical spot SNR: peak signal over background noise std.

    Background is estimated from the dimmer half of the pixels (inter-spot
    region); returns inf for a noiseless frame.
    """
    med = float(np.median(frame))
    bg = frame[frame < np.percentile(frame, 50)]
    noise = float(bg.std()) if bg.size else 0.0
    signal = float(frame.max()) - med
    return signal / noise if noise > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Estimator / turbulence theory references
# --------------------------------------------------------------------------- #
def zernike_variance_spectrum(coeffs_timeseries: np.ndarray) -> np.ndarray:
    """Per-mode variance across frames for a (n_frames, n_modes) coeff array."""
    coeffs = np.asarray(coeffs_timeseries, dtype=float)
    return coeffs.var(axis=0, ddof=1)


def noll_mode_variance_theory(j: int, pupil_diameter: float, r0: float) -> float:
    """Theoretical Kolmogorov variance [rad^2] of Noll mode j over the pupil.

    The variance carried by mode j alone is (Delta_{j-1} - Delta_j) * (D/r0)^(5/3),
    where Delta_J is Noll's cumulative residual after removing the first J modes.
    """
    d_over_r0_53 = (pupil_diameter / r0) ** (5.0 / 3.0)
    return (_noll_delta(j - 1) - _noll_delta(j)) * d_over_r0_53


def noll_mode_variance_spectrum(
    js, pupil_diameter: float, r0: float
) -> np.ndarray:
    """Theoretical per-mode variance for an iterable of Noll indices."""
    return np.array([noll_mode_variance_theory(j, pupil_diameter, r0) for j in js])


# --------------------------------------------------------------------------- #
# Phase-screen statistics
# --------------------------------------------------------------------------- #
def kolmogorov_structure_fn_theory(r: np.ndarray, r0: float) -> np.ndarray:
    """Kolmogorov phase structure function D_phi(r) = 6.88 (r/r0)^(5/3) [rad^2]."""
    r = np.asarray(r, dtype=float)
    return 6.88 * (r / r0) ** (5.0 / 3.0)


def structure_function(
    screen: np.ndarray, dx: float, max_lag: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical isotropic phase structure function of a screen.

    D_phi(r) = < [phi(x + r) - phi(x)]^2 >, averaged over the x and y shift
    directions and over the array. Returns (separations [m], D_phi [rad^2]) for
    integer pixel lags 1..max_lag.

    The FFT screen under-represents the lowest spatial frequencies, so the
    largest separations sit below the Kolmogorov curve (Lane et al. 1992); use
    mid-range separations for quantitative comparison.
    """
    s = np.asarray(screen, dtype=float)
    n = s.shape[0]
    if max_lag is None:
        max_lag = n // 2
    lags = np.arange(1, max_lag + 1)
    dphi = np.empty(lags.size)
    for i, k in enumerate(lags):
        diff_x = s[:, k:] - s[:, :-k]
        diff_y = s[k:, :] - s[:-k, :]
        sq = np.concatenate([diff_x.ravel() ** 2, diff_y.ravel() ** 2])
        dphi[i] = sq.mean()
    return lags * dx, dphi


def radial_psd(screen: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectral density of a phase screen.

    Returns (spatial frequency [cycles/m], radially-averaged PSD). For a
    Kolmogorov screen the mid-frequency PSD follows f^(-11/3).
    """
    s = np.asarray(screen, dtype=float)
    n = s.shape[0]
    psd2d = np.abs(np.fft.fftshift(np.fft.fft2(s))) ** 2
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    fxx, fyy = np.meshgrid(fx, fx)
    fr = np.sqrt(fxx ** 2 + fyy ** 2)

    # Radial binning onto the positive-frequency grid spacing.
    df = 1.0 / (n * dx)
    nbins = n // 2
    bins = np.floor(fr / df).astype(int)
    f_centres = (np.arange(nbins) + 0.5) * df
    psd_radial = np.empty(nbins)
    for b in range(nbins):
        sel = bins == b
        psd_radial[b] = psd2d[sel].mean() if sel.any() else np.nan
    return f_centres, psd_radial


# --------------------------------------------------------------------------- #
# Slope helpers for fit quality
# --------------------------------------------------------------------------- #
def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log(y) vs log(x); use to check power-law indices."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[good]), np.log(y[good]), 1)[0])
