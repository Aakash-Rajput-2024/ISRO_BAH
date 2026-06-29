"""Tier 3 -- real lab-data validation path.

Two groups:
  * grid auto-detection, validated NOW against a synthetic flat frame whose
    geometry is known exactly;
  * hardware checks (frame ingest, known-aberration injection, repeatability)
    that skip until real frames are dropped into tests/fixtures/realdata/.
"""
import glob
import os

import numpy as np
import pytest

from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame, render_frame
from validations import realdata as rd

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures", "realdata")


# --------------------------------------------------------------------------- #
# Runnable today: auto-detection against synthetic ground truth
# --------------------------------------------------------------------------- #
@pytest.mark.fast
def test_grid_autodetect_recovers_synthetic_geometry(pipe):
    """detect_grid recovers the known pitch and lenslet count of a flat frame."""
    cfg = pipe.cfg
    flat = flat_frame(cfg, pipe.geom)
    grid = rd.detect_grid(flat)
    assert abs(grid["pitch"] - cfg.pix_per_lenslet) <= 1.0
    assert grid["n_lenslets_x"] == cfg.n_lenslets
    assert grid["n_lenslets_y"] == cfg.n_lenslets
    # Grid origin near the first lenslet centre (~ppl/2).
    assert abs(grid["origin_x"] - cfg.pix_per_lenslet / 2.0) <= 1.5


@pytest.mark.fast
def test_load_frame_roundtrips_npy(tmp_path, pipe):
    """The .npy ingest path returns the same array (the fixture format)."""
    frame = flat_frame(pipe.cfg, pipe.geom)
    p = tmp_path / "frame.npy"
    np.save(p, frame)
    np.testing.assert_allclose(rd.load_frame(str(p)), frame)


# --------------------------------------------------------------------------- #
# Hardware: skipped until real fixtures exist
# --------------------------------------------------------------------------- #
def _have_realdata() -> bool:
    return os.path.isdir(FIXDIR) and bool(glob.glob(os.path.join(FIXDIR, "flat.*")))


realdata_only = pytest.mark.skipif(not _have_realdata(),
                                   reason="no real-data fixtures in tests/fixtures/realdata/")


@pytest.mark.hardware
@realdata_only
def test_calibration_grid_is_regular():
    """A real flat frame's detected grid is regular (uniform pitch)."""
    flat = rd.load_frame(glob.glob(os.path.join(FIXDIR, "flat.*"))[0])
    grid = rd.detect_grid(flat)
    spacings = np.diff(grid["cols"])
    assert spacings.std() / spacings.mean() < 0.05  # < 5% pitch jitter


@pytest.mark.hardware
@realdata_only
def test_known_defocus_injection_recovered():
    """Gold standard: a frame with a physically-injected known defocus
    recovers the expected Zernike coefficient.

    Expects fixtures: flat.* (calibration) and defocus.npy plus a sidecar
    defocus_truth.npy holding the commanded coefficient vector.
    """
    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(rd.load_frame(glob.glob(os.path.join(FIXDIR, "flat.*"))[0]))
    frame = rd.load_frame(os.path.join(FIXDIR, "defocus.npy"))
    truth = np.load(os.path.join(FIXDIR, "defocus_truth.npy"))
    out = pipe.process(frame)
    # Focus is Noll j=4 -> column index 2.
    np.testing.assert_allclose(out.coeffs[2], truth[2], atol=0.1)
