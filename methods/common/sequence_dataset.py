"""Load + window a temporal (N, T, D, D) dataset. Torch-free (numpy only).

A temporal model consumes a window of K consecutive frames [t-K+1 .. t] and
predicts a label at t+horizon (horizon=0 -> reconstruct the current step using
history; horizon>=1 -> forecast ahead for servo-lag compensation). Windows are
sliced lazily over memory-mapped arrays, so the full (N,T,D,D) cube is never
loaded into RAM.

Splitting is BY INSTANCE (the N axis): frames from one turbulence realisation
never straddle train/val/test, so there is no temporal leakage.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceData:
    frames: np.ndarray        # (N, T, D, D) memmap, uint8 or float16
    coeffs: np.ndarray        # (N, T, M)
    slopes: np.ndarray        # (N, T, 2V)
    wavefronts: np.ndarray    # (N, T, w, w)
    mean_frame: np.ndarray    # (N, D, D)
    manifest: dict

    @property
    def N(self) -> int: return self.frames.shape[0]

    @property
    def T(self) -> int: return self.frames.shape[1]

    def windows(self, K: int, horizon: int = 0, label: str = "coeffs",
                instances=None, normalize: bool = True) -> "WindowView":
        return WindowView(self, K, horizon, label, instances, normalize)


def load_sequences(path: str) -> SequenceData:
    """Memory-map a dataset folder produced by data/make_sequence_dataset.py."""
    def mm(name):
        return np.load(os.path.join(path, name), mmap_mode="r")
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    return SequenceData(mm("frames.npy"), mm("coeffs.npy"), mm("slopes.npy"),
                        mm("wavefronts.npy"), np.load(os.path.join(path, "mean_frame.npy")),
                        manifest)


class WindowView:
    """Indexable view producing (X, y) windows; safe to wrap in a torch Dataset."""

    def __init__(self, data: SequenceData, K: int, horizon: int, label: str,
                 instances, normalize: bool):
        if label not in ("coeffs", "slopes", "wavefronts"):
            raise ValueError("label must be coeffs | slopes | wavefronts")
        self.data, self.K, self.horizon, self.label = data, K, horizon, label
        self.normalize = normalize
        T = data.T
        if instances is None:
            instances = range(data.N)
        # Valid target index t: needs K frames of history and t+horizon < T.
        self.pairs = [(i, t) for i in instances
                      for t in range(K - 1, T - horizon)]
        # Normalisation scale for uint8/float frames.
        self._scale = 255.0 if data.frames.dtype == np.uint8 else 1.0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        i, t = self.pairs[idx]
        x = np.asarray(self.data.frames[i, t - self.K + 1:t + 1], dtype=np.float32)
        if self.normalize and self._scale != 1.0:
            x = x / self._scale
        y = np.asarray(getattr(self.data, self.label)[i, t + self.horizon],
                       dtype=np.float32)
        return x, y                       # X: (K, D, D)   y: label at t+horizon


def instance_split(n: int, fracs=(0.6, 0.2, 0.2), seed: int = 0) -> dict:
    """Disjoint train/val/test instance-index lists (by the N axis)."""
    idx = np.random.default_rng(seed).permutation(n).tolist()
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va],
            "test": idx[n_tr + n_va:]}


def split_from_manifest(data: SequenceData) -> dict:
    """The by-instance split recorded at generation time."""
    return data.manifest["split"]
