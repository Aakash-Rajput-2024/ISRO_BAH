"""Shared pytest fixtures and tolerance config for the validation suite.

Markers (see pytest.ini):
  fast      -- < ~10 s total; runs on every push / `./run.sh test`
  slow      -- Monte-Carlo characterisation; nightly
  crossval  -- needs an independent library (aotools / hcipy)
  hardware  -- needs real lab-data fixtures
  golden    -- C-port oracle conformance
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame
from validations.thresholds import TOL  # canonical, shared with the report


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture(scope="session")
def pipe(cfg) -> WFSPipeline:
    """A pipeline calibrated on a noiseless flat frame (geometric references)."""
    p = WFSPipeline(cfg)
    p.calibrate(flat_frame(p.cfg, p.geom))
    return p


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG so every test is reproducible."""
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def tol() -> dict:
    return TOL
