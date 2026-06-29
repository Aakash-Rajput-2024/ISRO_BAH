"""Tier 4 -- golden-vector conformance.

Asserts the current Python pipeline still reproduces the committed golden
vectors (catches accidental drift in the reference oracle), and that the stored
config matches the live Config. The same fixtures + `oracle.compare` are what a
C-port test harness will use to prove numerical equivalence.
"""
import numpy as np
import pytest

from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame
from validations import oracle


@pytest.fixture(scope="module")
def golden():
    try:
        return oracle.load_golden()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.mark.golden
def test_config_matches_manifest(golden):
    """Live Config must match the config the fixtures were generated under."""
    _, manifest = golden
    assert manifest["config_hash"] == oracle.config_hash(Config())


@pytest.mark.golden
def test_centroiding_reproduces_golden(golden, tol):
    """frame -> centroids reproduces the stored vectors (noiseless)."""
    arrays, _ = golden
    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(arrays["flat_frame"])
    cx, cy, _ = pipe.slopes_from_frame(arrays["frame"])
    assert oracle.compare(cx, arrays["cx"], atol=tol["golden_centroid_px"])["ok"]
    assert oracle.compare(cy, arrays["cy"], atol=tol["golden_centroid_px"])["ok"]


@pytest.mark.golden
def test_reconstruction_reproduces_golden(golden, tol):
    """slopes -> coeffs reproduces the stored vectors using the STORED R.

    Uses the committed R (not a fresh pinv) so the check is BLAS-independent --
    exactly the contract the C port must satisfy.
    """
    arrays, _ = golden
    coeffs = arrays["R"] @ arrays["slopes"]
    rep = oracle.compare(coeffs, arrays["coeffs"], rtol=tol["golden_coeff_rel"],
                         atol=1e-12)
    assert rep["ok"], rep


@pytest.mark.golden
def test_operators_unchanged(golden):
    """The interaction matrix D rebuilt from geometry matches the stored D."""
    arrays, _ = golden
    pipe = WFSPipeline(Config())
    np.testing.assert_allclose(pipe.recon.D, arrays["D"], rtol=1e-12, atol=1e-12)
