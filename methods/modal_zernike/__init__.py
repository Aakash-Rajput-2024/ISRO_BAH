"""Method 1 -- Modal (Zernike) wavefront reconstruction.

Writes the wavefront as W = sum_k a_k Z_k and solves the slopes-to-coefficients
inverse as a single matrix-vector product a = pinv(D) @ s. Re-exports the shared
`Config`/`Geometry` for convenience so callers can do:

    from methods.modal_zernike import Config, WFSPipeline
"""

from ..common.config import Config
from ..common.geometry import Geometry
from .reconstruct import ModalReconstructor
from .pipeline import WFSPipeline

__all__ = ["Config", "Geometry", "ModalReconstructor", "WFSPipeline"]
