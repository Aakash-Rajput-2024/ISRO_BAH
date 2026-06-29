"""Load a synthetic dataset (data/synthetic/<name>) for ResU-Net training.

Pairs each detector frame with a coarse, [-1, 1]-scaled wavefront target:
  - input:  average-added, [0,1]-normalised frame  -> (1, npix, npix)
  - target: pupil wavefront area-downsampled to (target, target), divided by a
            robust amplitude scale and clipped to [-1, 1] (matches the tanh head)

The amplitude scale and the mean frame are returned so inference can invert the
scaling and reproduce the average-added preprocessing.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import average_added


def load_synthetic(path: str, target: int = 48, avg_added: bool = True):
    """Return (X, Y, meta) tensors/dict for the dataset folder at `path`."""
    frames = np.load(os.path.join(path, "frames.npy")).astype(np.float32)
    wavefronts = np.load(os.path.join(path, "wavefronts.npy")).astype(np.float32)
    n, npix, _ = frames.shape

    ref = frames.mean(axis=0) if avg_added else None
    X = np.stack([average_added(f, ref) for f in frames])[:, None, :, :]

    wf = torch.from_numpy(wavefronts)[:, None, :, :]
    wf_coarse = F.adaptive_avg_pool2d(wf, (target, target))  # area downsample
    scale = float(np.percentile(np.abs(wf_coarse.numpy()), 99.5)) or 1.0
    Y = torch.clamp(wf_coarse / scale, -1.0, 1.0)

    meta = {
        "npix": int(npix),
        "target": int(target),
        "wf_scale": scale,
        "avg_added": bool(avg_added),
        "mean_frame": ref,  # np.ndarray or None
        "n": int(n),
    }
    if os.path.exists(os.path.join(path, "manifest.json")):
        with open(os.path.join(path, "manifest.json")) as f:
            meta["source_manifest"] = json.load(f)

    return torch.from_numpy(X).float(), Y.float(), meta


def split(X, Y, val_frac: float = 0.2, seed: int = 0):
    """Deterministic train/val split."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    vi, ti = idx[:n_val], idx[n_val:]
    return (X[ti], Y[ti]), (X[vi], Y[vi])
