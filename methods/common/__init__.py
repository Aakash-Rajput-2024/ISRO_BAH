"""Method-agnostic core shared by every reconstruction method.

  config       -- system parameters (MLA, detector, pupil)
  geometry     -- pupil grid, valid sub-apertures, averaging operator
  zernike      -- Noll-ordered Zernike polynomials
  phasescreen  -- Kolmogorov phase screens + frozen-flow time series
  simulate     -- synthetic SH-WFS frame generator (the test oracle)
  centroid     -- thresholded centre-of-gravity spot centroiding
  turbulence   -- r0 (Fried) and tau0 (coherence time) estimators
"""

from .config import Config
from .geometry import Geometry

__all__ = ["Config", "Geometry"]
