"""Shack-Hartmann wavefront sensing toolkit (synthetic prototype).

Pipeline: SH-WFS frame -> centroids -> slopes -> wavefront / Zernike
coefficients -> turbulence parameters (r0, tau0) -> DM actuator map.

This package is the Python reference implementation. The performance-critical
inner loop (centroiding + matrix-vector reconstruction) is intended to be
ported to C later; this code is the ground-truth oracle to validate that port.
"""

from .config import Config
from .geometry import Geometry
from .pipeline import WFSPipeline

__all__ = ["Config", "Geometry", "WFSPipeline"]
