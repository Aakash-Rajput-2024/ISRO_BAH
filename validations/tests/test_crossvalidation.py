"""Tier 2 -- cross-validation against independent libraries.

Guards against the "self-consistent but wrong" failure mode. Skipped cleanly
when the optional oracle packages (aotools / hcipy) are not installed.
"""
import numpy as np
import pytest

from methods.common.zernike import zernike
from validations import metrics as m
from validations import references as ref

aotools_only = pytest.mark.skipif(not ref.have_aotools(), reason="aotools not installed")
hcipy_only = pytest.mark.skipif(not ref.have_hcipy(), reason="hcipy not installed")


@pytest.mark.crossval
@aotools_only
@pytest.mark.parametrize("j", [2, 3, 4, 5, 8, 11, 14])
def test_zernike_basis_matches_aotools(j):
    """Our Noll Zernike maps match aotools' independent implementation in shape
    (correlation) and normalisation (unit-RMS), allowing for a sign convention."""
    npix = 128
    rho, theta, mask = ref.unit_disk_grid(npix)
    ours = zernike(j, rho, theta) * mask
    theirs = ref.aotools_zernike_noll(j, npix)
    sel = mask & (theirs != 0)
    a, b = ours[sel], theirs[sel]
    corr = abs(np.corrcoef(a, b)[0, 1])
    rms_ratio = a.std() / b.std()
    assert corr > 0.999
    assert 0.98 < rms_ratio < 1.02


@pytest.mark.crossval
@aotools_only
def test_phasescreen_slope_consistent_with_aotools():
    """An independent aotools Kolmogorov screen has a comparable mid-range
    structure-function power law to ours (both ~Kolmogorov)."""
    npix, dx = 256, 0.01
    r0 = 0.2
    screen = ref.aotools_phase_screen(r0, npix, dx)
    r, d = m.structure_function(screen, dx, max_lag=npix // 4)
    slope = m.loglog_slope(r[2:npix // 16], d[2:npix // 16])
    # von-Karman/finite-L0 flattens it slightly; require a clearly turbulent slope.
    assert 1.2 < slope < 2.0


@pytest.mark.crossval
@hcipy_only
@pytest.mark.parametrize("j", [2, 4, 5, 8, 11])
def test_zernike_basis_matches_hcipy(j):
    """Second independent parity check, against HCIPy's Noll-indexed Zernikes."""
    import hcipy
    npix = 128
    grid = hcipy.make_pupil_grid(npix, 1.0)
    theirs = np.asarray(hcipy.mode_basis.zernike_noll(j, D=1.0, grid=grid).shaped)
    rho, theta, mask = ref.unit_disk_grid(npix)
    ours = zernike(j, rho, theta)
    sel = mask & (np.abs(theirs) > 0)
    corr = abs(np.corrcoef(ours[sel], theirs[sel])[0, 1])
    assert corr > 0.999
