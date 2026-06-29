"""Method 3 -- Deep-learning (ResU-Net) wavefront reconstruction.

PyTorch reimplementation of Noel et al. (2023): a Res-UNet CNN maps a raw
SH-WFS frame directly to a (zonal) wavefront, bypassing centroiding and matrix
inversion. See README.md for the faithful mapping to the paper and the two
adaptations (PyTorch instead of TensorFlow; trained on the synthetic atmospheric
dataset since the paper's wind-tunnel data is unavailable).

    from methods.deep_resunet import ResUNet, ResUNetReconstructor
"""

from .model import ResUNet, pick_device
from .reconstruct import ResUNetReconstructor, DeepResult

__all__ = ["ResUNet", "pick_device", "ResUNetReconstructor", "DeepResult"]
