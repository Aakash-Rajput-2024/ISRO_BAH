"""ResU-Net for direct SH-WFS frame -> wavefront reconstruction (Method 3).

PyTorch reimplementation of the network in Noel et al. (2023), "Shack-Hartmann
wavefront reconstruction by deep learning neural network for adaptive optics,"
Proc. SPIE 12693, 126930G.

Faithful elements (paper Fig. 3):
  - Multi-kernel residual blocks: parallel 3x3/5x5/7x7/9x9 convolutions (each
    out_ch/4 features) concatenated with the input, fused by a 1x1 conv, then
    batch-norm + a residual add with a 1x1 shortcut, ReLU.
  - Encoder downsamples with stride-4 convolutions (not max-pool), preserving
    spatial information for the average-added input; features double each level.
  - Decoder upsamples with stride-4 transposed convolutions; encoder features
    are concatenated into the matching decoder level (U-Net skips).
  - A fully-connected head with a tanh activation maps to the (zonal) wavefront,
    which can be negative -- hence tanh -- and is resized to the target grid.

The network is deliberately zonal (outputs a wavefront map, not Zernike modes),
matching the paper's argument against modal reconstruction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiKernelResBlock(nn.Module):
    """Residual block with parallel multi-scale convolution branches."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        q = out_ch // 4
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_ch, q, k, padding=k // 2) for k in (3, 5, 7, 9)]
        )
        self.fuse = nn.Conv2d(q * 4 + in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = torch.cat([branch(x) for branch in self.branches], dim=1)
        b = torch.cat([b, x], dim=1)          # concatenate with the input
        b = self.fuse(b)                       # 1x1 fuse
        return self.act(self.bn(b) + self.shortcut(x))


def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize x to ref's spatial size (guards against off-by-one at skips)."""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="nearest")
    return x


class ResUNet(nn.Module):
    """Res-UNet: raw frame (1xHxW) -> coarse wavefront map (1 x target x target)."""

    def __init__(self, target: int = 48, base: int = 24, adapt: int = 8):
        super().__init__()
        self.target = target
        b1, b2, b3 = base, base * 2, base * 4

        self.enc0 = MultiKernelResBlock(1, b1)
        self.down1 = nn.Conv2d(b1, b2, kernel_size=4, stride=4)
        self.enc1 = MultiKernelResBlock(b2, b2)
        self.down2 = nn.Conv2d(b2, b3, kernel_size=4, stride=4)
        self.bott = MultiKernelResBlock(b3, b3)

        self.up1 = nn.ConvTranspose2d(b3, b2, kernel_size=4, stride=4)
        self.dec1 = MultiKernelResBlock(b2 + b2, b2)
        self.up0 = nn.ConvTranspose2d(b2, b1, kernel_size=4, stride=4)
        self.dec0 = MultiKernelResBlock(b1 + b1, b1)

        self.pool = nn.AdaptiveAvgPool2d((adapt, adapt))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(b1 * adapt * adapt, target * target),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        b = self.bott(self.down2(e1))
        d1 = self.dec1(torch.cat([_match(self.up1(b), e1), e1], dim=1))
        d0 = self.dec0(torch.cat([_match(self.up0(d1), e0), e0], dim=1))
        y = self.head(self.pool(d0))
        return y.view(-1, 1, self.target, self.target)


def pick_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
