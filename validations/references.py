"""Lazy adapters to independent AO libraries for Tier-2 cross-validation.

These guard against the "self-consistent but wrong" failure mode: our forward
and inverse models share an averaging operator, so a shared bug (e.g. a Zernike
normalisation or sign error) would pass the round-trip tests. Comparing against
an independently-written implementation catches that.

Nothing here is imported at module load; callers check `have_aotools()` /
`have_hcipy()` first and skip when absent, so the core suite never hard-depends
on these packages.
"""
from __future__ import annotations

import math

import numpy as np


def _ensure_numpy_math() -> None:
    """aotools<=1.0.7 calls numpy.math.factorial, removed in NumPy 2.x.

    Restore the alias to the stdlib math module so the package imports/runs.
    """
    if not hasattr(np, "math"):
        np.math = math  # type: ignore[attr-defined]


def have_aotools() -> bool:
    try:
        _ensure_numpy_math()
        import aotools  # noqa: F401
        return True
    except Exception:
        return False


def have_hcipy() -> bool:
    try:
        import hcipy  # noqa: F401
        return True
    except Exception:
        return False


def unit_disk_grid(npix: int):
    """(rho, theta, mask) on the same convention as methods.common.geometry.Geometry."""
    ax = np.arange(npix) + 0.5 - npix / 2.0
    xx, yy = np.meshgrid(ax, ax)
    rho = np.sqrt(xx ** 2 + yy ** 2) / (npix / 2.0)
    theta = np.arctan2(yy, xx)
    return rho, theta, rho <= 1.0


def aotools_zernike_noll(j: int, npix: int) -> np.ndarray:
    """Independent Noll-ordered, unit-RMS Zernike map from aotools."""
    _ensure_numpy_math()
    from aotools.functions import zernike as az
    return np.asarray(az.zernike_noll(j, npix))


def aotools_phase_screen(r0: float, npix: int, dx: float, L0: float = 100.0) -> np.ndarray:
    """Independent Kolmogorov/von-Karman phase screen from aotools.

    Large L0 approximates pure Kolmogorov. Returned in radians.
    """
    _ensure_numpy_math()
    from aotools import turbulence as at
    return np.asarray(at.ft_phase_screen(r0, npix, dx, L0, 0.001))
