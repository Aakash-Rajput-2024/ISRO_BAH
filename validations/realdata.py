"""Real lab-data ingest + sub-aperture grid auto-detection (Tier 3).

This is the bridge from synthetic validation to hardware frames. The grid
auto-detection is testable *today* against a synthetic flat frame (known truth);
the frame loader and the known-aberration-injection checks run once real
fixtures are dropped into tests/fixtures/realdata/.

Data contract (see docs/VALIDATION.md):
  - a flat-wavefront calibration frame + science frames, same resolution
  - grayscale intensity, background not subtracted
  - square detector; one microlens array spanning the pupil
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import find_peaks


def load_frame(path: str) -> np.ndarray:
    """Load a detector frame as a 2-D float array.

    Supports .npy directly; image formats (.bmp/.png/.tif) via Pillow if
    available; .fits via astropy if available. Raises a clear error otherwise.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(float)
    if ext in (".fits", ".fit"):
        from astropy.io import fits  # optional
        with fits.open(path) as hdul:
            return np.asarray(hdul[0].data, dtype=float)
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"loading {ext} frames needs Pillow (pip install pillow), or convert to .npy"
        ) from e
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def estimate_pitch(image: np.ndarray, min_pitch: int = 4) -> float:
    """Estimate the lenslet pitch [px] from the spot grid via autocorrelation.

    Robust to not knowing the lenslet count: the first autocorrelation peak of
    the column-summed intensity is the inter-spot spacing.
    """
    proj = image.sum(axis=0).astype(float)
    proj -= proj.mean()
    ac = np.correlate(proj, proj, mode="full")[proj.size - 1:]
    peaks, _ = find_peaks(ac, distance=min_pitch)
    if peaks.size == 0:
        raise ValueError("no periodicity found; is this a spot-grid frame?")
    return float(peaks[0])


def detect_spot_centers(projection: np.ndarray, pitch: float):
    """Spot centres along one axis from an intensity projection."""
    proj = projection.astype(float)
    proj = proj - proj.min()
    peaks, _ = find_peaks(proj, distance=max(3, int(pitch * 0.5)))
    return peaks


def detect_grid(flat_image: np.ndarray) -> dict:
    """Auto-detect the sub-aperture grid from a flat-wavefront frame.

    Returns dict with detected pitch [px], per-axis spot-centre arrays, the
    inferred lenslet count, and the grid origin (first centre on each axis).
    """
    pitch = estimate_pitch(flat_image)
    cols = detect_spot_centers(flat_image.sum(axis=0), pitch)
    rows = detect_spot_centers(flat_image.sum(axis=1), pitch)
    return {
        "pitch": pitch,
        "cols": cols,
        "rows": rows,
        "n_lenslets_x": int(cols.size),
        "n_lenslets_y": int(rows.size),
        "origin_x": float(cols[0]) if cols.size else float("nan"),
        "origin_y": float(rows[0]) if rows.size else float("nan"),
    }
