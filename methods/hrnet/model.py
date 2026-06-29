"""Method 4 -- HRNet-style network with progressive temporal injection.

Baseline architecture only: this file defines the model. Data flow, dataset
wiring, training, and inference are handled separately.

Temporal scheme
---------------
Inputs are frames  t, t-1, ..., t-(n_frames-1)  (current frame first); for
n_frames=5 that is t plus the previous 4. The target is the wavefront at t.

    stem(t) -> [block1] -> +t-1 -> [block2] -> +t-2 -> [block3] -> ... -> head

  * frame t seeds the network;
  * after each "big HRNet block" (one stage = ResNet blocks per branch +
    cross-scale fusion) the next older frame is injected (added) into the
    highest-resolution branch;
  * a SKIP connection feeds the *previous block's* branch-0 input into the
    current block (rolling chain) -- NOT the global start broadcast everywhere.
    Block 2 therefore has no skip (its predecessor's start is the global start,
    which is excluded); blocks 3.. skip from their immediate predecessor.

`n_frames` is the only temporal knob; it is decoupled from `n_branches` (the
number of parallel resolution streams, which grow HRNet-style over the stages).

I/O contract
------------
  input : (B, n_frames, H, W) -- channel 0 = t, channel i = t-i
  output: head="map"    -> (B, 1, out_size, out_size) wavefront at t, tanh [-1,1]
          head="coeffs" -> (B, n_coeffs) Zernike coefficients at t
Size-agnostic (any H, W; ideally divisible by 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


def _stem(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two stride-2 convs -> 1/4 resolution, `out_ch` wide (the HRNet stem)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
        nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False),
    )


class BasicBlock(nn.Module):
    """ResNet basic block at constant channel count."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = conv3x3(ch, ch)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = conv3x3(ch, ch)
        self.bn2 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


def _branch(ch: int, n_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[BasicBlock(ch) for _ in range(n_blocks)])


class FusionModule(nn.Module):
    """Cross-resolution fusion: out_i = ReLU( sum_j transform(x_j -> branch i) )."""

    def __init__(self, channels):
        super().__init__()
        n = len(channels)
        self.n = n
        self.paths = nn.ModuleList()
        for i in range(n):
            row = nn.ModuleList()
            for j in range(n):
                if j == i:
                    row.append(None)
                elif j > i:                      # lower-res source -> upsample
                    row.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i])))
                else:                            # higher-res source -> downsample
                    convs, cin = [], channels[j]
                    for step in range(i - j):
                        last = step == i - j - 1
                        cout = channels[i] if last else cin
                        convs += [conv3x3(cin, cout, stride=2), nn.BatchNorm2d(cout)]
                        if not last:
                            convs.append(nn.ReLU(inplace=False))
                        cin = cout
                    row.append(nn.Sequential(*convs))
            self.paths.append(row)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, xs):
        out = []
        for i in range(self.n):
            y = xs[i]
            for j in range(self.n):
                if j == i:
                    continue
                t = self.paths[i][j](xs[j])
                if j > i:
                    t = F.interpolate(t, size=xs[i].shape[-2:], mode="bilinear",
                                      align_corners=False)
                y = y + t
            out.append(self.relu(y))
        return out


class HRStage(nn.Module):
    """One 'big block': ResNet blocks per branch, then cross-branch fusion."""

    def __init__(self, channels, n_blocks):
        super().__init__()
        self.branches = nn.ModuleList([_branch(c, n_blocks) for c in channels])
        self.fuse = FusionModule(channels)

    def forward(self, xs):
        xs = [branch(x) for branch, x in zip(self.branches, xs)]
        return self.fuse(xs)


class HRNetWavefront(nn.Module):
    def __init__(self, n_frames: int = 5, base_channels: int = 18,
                 n_branches: int = 3, blocks_per_branch: int = 2,
                 frame_depth_growth: int = 1,
                 head: str = "map", out_size: int = 48, n_coeffs: int = 20):
        super().__init__()
        if head not in ("map", "coeffs"):
            raise ValueError("head must be 'map' or 'coeffs'")
        self.n_frames = n_frames                  # frames t .. t-(n_frames-1)
        self.head_type = head
        self.out_size = out_size

        channels = [base_channels * (2 ** k) for k in range(n_branches)]
        self.channels = channels
        self.branches_at = [min(s + 1, n_branches) for s in range(n_frames)]

        # Per-frame encoder = stem + residual blocks. Later-injected frames pass
        # through fewer main blocks, so their encoder is made DEEPER to
        # compensate (depth grows with the lag s; the last frame is deepest).
        self.frame_encoders = nn.ModuleList([
            nn.Sequential(_stem(1, channels[0]),
                          *[BasicBlock(channels[0])
                            for _ in range(blocks_per_branch + frame_depth_growth * s)])
            for s in range(n_frames)])
        # LayerNorm (== GroupNorm with 1 group, size-agnostic) after each
        # temporal connection (injection + skip merge), for stages 1..n-1.
        self.merge_norms = nn.ModuleList([
            nn.GroupNorm(1, channels[0]) for _ in range(n_frames - 1)])

        self.transitions = nn.ModuleDict()        # spawn a branch when count grows
        self.stages = nn.ModuleList()
        for s in range(n_frames):
            if s > 0 and self.branches_at[s] > self.branches_at[s - 1]:
                nb = self.branches_at[s] - 1
                self.transitions[str(s)] = nn.Sequential(
                    conv3x3(channels[nb - 1], channels[nb], stride=2),
                    nn.BatchNorm2d(channels[nb]), nn.ReLU(inplace=False))
            self.stages.append(HRStage(channels[:self.branches_at[s]], blocks_per_branch))

        final_b = self.branches_at[-1]
        hc = base_channels * 2
        self.fuse_head = nn.Sequential(
            nn.Conv2d(sum(channels[:final_b]), hc, 1, bias=False),
            nn.BatchNorm2d(hc), nn.ReLU(inplace=False))
        if head == "map":
            self.head = nn.Sequential(
                conv3x3(hc, hc), nn.BatchNorm2d(hc), nn.ReLU(inplace=False),
                nn.Conv2d(hc, 1, kernel_size=1))
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hc, n_coeffs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.n_frames:
            raise ValueError(f"expected {self.n_frames} frames on dim 1, got {x.shape[1]}")
        frames = [x[:, i:i + 1] for i in range(self.n_frames)]   # frame i = t-i

        branches = [self.frame_encoders[0](frames[0])]           # seed with t
        branches = self.stages[0](branches)                      # block 1
        prev_block_start = None                                  # rolling skip source
        for s in range(1, self.n_frames):
            if str(s) in self.transitions:                       # grow a branch
                branches = branches + [self.transitions[str(s)](branches[-1])]
            cur = branches[0] + self.frame_encoders[s](frames[s])  # inject t-s (deep)
            if prev_block_start is not None:                     # skip from PREVIOUS
                cur = cur + prev_block_start                     # block's start (not global)
            cur = self.merge_norms[s - 1](cur)                   # LayerNorm after connections
            branches[0] = cur
            prev_block_start = cur                               # becomes next block's skip
            branches = self.stages[s](branches)                 # big block

        size = branches[0].shape[-2:]
        ups = [branches[0]] + [
            F.interpolate(b, size=size, mode="bilinear", align_corners=False)
            for b in branches[1:]]
        feat = self.fuse_head(torch.cat(ups, dim=1))

        if self.head_type == "map":
            y = self.head(feat)
            y = F.interpolate(y, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
            return torch.tanh(y)
        return self.head(feat)


def pick_device() -> torch.device:
    """Prefer Apple MPS, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
