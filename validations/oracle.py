"""Golden-vector oracle for the C-port (Tier 4).

The Python pipeline is the reference implementation. This module pins canonical
inputs/outputs for the performance-critical inner loop (centroiding -> slopes ->
modal reconstruction) so the eventual C port can be proven numerically
equivalent against the *same* fixtures.

Key choices:
  - The reconstructor matrices D and R are stored explicitly. `pinv`/SVD is not
    bit-stable across BLAS builds, so the C port must consume the stored R
    rather than recompute it; comparisons of `R @ slopes` use the stored R.
  - A manifest records every Config field plus a hash, so fixtures and config
    cannot silently drift apart.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np

from methods.common.config import Config

GOLDEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tests", "fixtures", "golden",
)
VECTORS_NPZ = "vectors.npz"
MANIFEST_JSON = "manifest.json"


def config_dict(cfg: Config) -> dict:
    """Flat dict of Config fields plus the derived quantities a C port needs."""
    d = dataclasses.asdict(cfg)
    d.update(
        npix=cfg.npix,
        pupil_diameter=cfg.pupil_diameter,
        pupil_dx=cfg.pupil_dx,
        slope_scale=cfg.slope_scale,
    )
    return d


def config_hash(cfg: Config) -> str:
    """Stable hash of the config so drift between fixtures and code is caught."""
    blob = json.dumps(config_dict(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def dump_golden(arrays: dict, cfg: Config, directory: str = GOLDEN_DIR) -> None:
    """Write golden vectors (.npz) and a manifest (.json) into `directory`."""
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, VECTORS_NPZ), **arrays)
    manifest = {
        "config": config_dict(cfg),
        "config_hash": config_hash(cfg),
        "arrays": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
        "tolerances": {
            "centroid_px": 1e-6,
            "slope_rel": 1e-9,
            "coeff_rel": 1e-9,
        },
        "note": "Reference vectors for the SH-WFS C port. Compare against the "
                "stored R (do not recompute pinv).",
    }
    with open(os.path.join(directory, MANIFEST_JSON), "w") as f:
        json.dump(manifest, f, indent=2)


def load_golden(directory: str = GOLDEN_DIR):
    """Return (arrays_dict, manifest_dict). Raises if fixtures are absent."""
    npz_path = os.path.join(directory, VECTORS_NPZ)
    man_path = os.path.join(directory, MANIFEST_JSON)
    if not (os.path.exists(npz_path) and os.path.exists(man_path)):
        raise FileNotFoundError(
            f"golden fixtures missing in {directory}; run scripts/gen_golden_vectors.py"
        )
    arrays = dict(np.load(npz_path))
    with open(man_path) as f:
        manifest = json.load(f)
    return arrays, manifest


def compare(got: np.ndarray, expected: np.ndarray, atol: float = 0.0,
            rtol: float = 0.0) -> dict:
    """Tolerance comparison the C-port test harness can reuse.

    Returns a report dict with max abs/rel error and a pass flag. `rtol` is
    measured against the expected magnitude (eps-guarded).
    """
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    abs_err = np.abs(got - expected)
    denom = np.maximum(np.abs(expected), 1e-30)
    rel_err = abs_err / denom
    ok = bool((abs_err <= atol + rtol * denom).all())
    return {
        "ok": ok,
        "max_abs": float(abs_err.max()),
        "max_rel": float(rel_err.max()),
        "atol": atol,
        "rtol": rtol,
    }
