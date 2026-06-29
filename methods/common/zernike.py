"""Zernike polynomials in Noll ordering, normalised so each mode has unit
variance over the unit disk (Noll 1976 convention).

Used both to build the modal interaction matrix (slopes per unit coefficient)
and to reconstruct a wavefront map from a coefficient vector.
"""
from __future__ import annotations

import math

import numpy as np


def noll_to_nm(j: int) -> tuple[int, int]:
    """Map a Noll single index j (>=1) to radial/azimuthal orders (n, m).

    Sign of m encodes parity: m > 0 -> cosine term, m < 0 -> sine term.
    """
    if j < 1:
        raise ValueError("Noll index j must be >= 1")
    n = 0
    j1 = j - 1
    while j1 > n:
        n += 1
        j1 -= n
    m = (-1) ** j * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2))
    return n, m


def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Radial polynomial R_n^m(rho); m is non-negative here."""
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        R += c * rho ** (n - 2 * k)
    return R


def zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate the j-th (Noll) Zernike polynomial on a polar grid.

    rho, theta are arrays; values for rho > 1 are returned as-is (the caller is
    expected to mask the pupil). Normalisation: unit variance over the disk.
    """
    n, m = noll_to_nm(j)
    am = abs(m)
    R = _radial(n, am, rho)
    if m > 0:
        return math.sqrt(n + 1) * R * math.sqrt(2) * np.cos(am * theta)
    if m < 0:
        return math.sqrt(n + 1) * R * math.sqrt(2) * np.sin(am * theta)
    return math.sqrt(n + 1) * R


def zernike_basis(
    js, rho: np.ndarray, theta: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Stack of Zernike maps for the given Noll indices.

    Returns an array of shape (len(js), *rho.shape). Outside the pupil mask the
    maps are set to zero so they can be linearly combined safely.
    """
    maps = np.stack([zernike(j, rho, theta) for j in js], axis=0)
    if mask is not None:
        maps = maps * mask[None, :, :]
    return maps
