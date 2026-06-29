"""Turbulence characterisation from a time series of reconstructed wavefronts.

Fried parameter r0
------------------
For Kolmogorov turbulence the variance of Zernike coefficient j (in rad^2) scales
as (D/r0)^(5/3). Noll (1976) gives the residual wavefront variance after the
first J modes are removed, in units of (D/r0)^(5/3):

    Delta_J  (Noll Table IV) -- tabulated below for small J, asymptotic beyond.

The variance captured by modes j_start..J equals
    (Delta_{j_start-1} - Delta_J) * (D/r0)^(5/3),
so measuring that captured variance gives (D/r0)^(5/3) and hence r0. We start at
j_start = 4 (skip piston/tip/tilt) to reduce sensitivity to the FFT screen's
low-frequency bias.

Coherence time tau0
-------------------
The temporal autocorrelation of the wavefront decays as the screen blows across
the pupil; the 1/e decay time is a data-driven coherence time. For comparison
the textbook relation tau0 = 0.314 * r0 / v is also reported when v is known.
"""
from __future__ import annotations

import numpy as np

# Noll cumulative residual variance Delta_J in units of (D/r0)^(5/3).
_NOLL_DELTA = {
    1: 1.0299, 2: 0.582, 3: 0.134, 4: 0.111, 5: 0.0880, 6: 0.0648,
    7: 0.0587, 8: 0.0525, 9: 0.0463, 10: 0.0401, 11: 0.0377, 12: 0.0352,
    13: 0.0328, 14: 0.0304, 15: 0.0279, 16: 0.0267, 17: 0.0255, 18: 0.0243,
    19: 0.0232, 20: 0.0220, 21: 0.0208,
}


def _noll_delta(j: int) -> float:
    if j in _NOLL_DELTA:
        return _NOLL_DELTA[j]
    # Asymptotic residual for large J (Noll 1976).
    return 0.2944 * j ** (-np.sqrt(3.0) / 2.0)


def estimate_r0(
    coeffs_timeseries: np.ndarray,
    pupil_diameter: float,
    j_start: int = 4,
) -> float:
    """Estimate r0 [m] from a (n_frames, n_modes) array of Zernike coefficients.

    Columns are Noll modes j = 2, 3, ..., n_modes+1 (piston excluded).
    """
    coeffs = np.asarray(coeffs_timeseries)
    n_modes = coeffs.shape[1]
    j_last = n_modes + 1  # Noll index of the final column

    var = coeffs.var(axis=0, ddof=1)  # variance per mode [rad^2]
    # Column index of Noll mode j is (j - 2).
    captured = var[(j_start - 2):].sum()

    coeff = _noll_delta(j_start - 1) - _noll_delta(j_last)
    if coeff <= 0 or captured <= 0:
        raise ValueError("invalid mode range for r0 estimation")

    d_over_r0_53 = captured / coeff           # (D/r0)^(5/3)
    r0 = pupil_diameter / d_over_r0_53 ** (3.0 / 5.0)
    return float(r0)


def estimate_tau0(
    wavefronts: np.ndarray,
    dt: float,
) -> float:
    """Data-driven coherence time [s]: 1/e decay of the temporal autocorrelation.

    `wavefronts` is (n_frames, n_points) -- e.g. flattened pupil phase samples or
    a slope/coefficient time series. Returns the lag at which the mean normalised
    autocorrelation first drops below 1/e (linearly interpolated).
    """
    x = np.asarray(wavefronts, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    var0 = (x * x).sum(axis=1).mean()
    if var0 <= 0:
        return float("nan")

    max_lag = n // 2
    ac = np.empty(max_lag)
    for lag in range(max_lag):
        prod = (x[: n - lag] * x[lag:]).sum(axis=1)
        ac[lag] = prod.mean() / var0

    target = np.exp(-1.0)
    below = np.where(ac < target)[0]
    if below.size == 0:
        return float(max_lag * dt)
    k = below[0]
    if k == 0:
        return 0.0
    # Linear interpolation between lag k-1 (>=target) and k (<target).
    a, b = ac[k - 1], ac[k]
    frac = (a - target) / (a - b) if a != b else 0.0
    return float((k - 1 + frac) * dt)
