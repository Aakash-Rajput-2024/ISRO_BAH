"""C++ port of the classical SH-WFS inner loop (centroiding + slopes).

Mirrors methods/common/centroid.py (thresholded centre-of-gravity) and the
slope step of methods/modal_zernike/pipeline.py, but runs the per-sub-aperture
pixel loop in compiled C++ instead of numpy -- the step the README earmarks for
a C port and the dominant per-frame cost of the modal method.

The shared library is loaded via ctypes and auto-compiled with the system C++
compiler on first import (and rebuilt whenever centroid.cpp changes), so there
is no extra Python dependency and no manual build step. If no compiler is
available, importing the callables raises CppUnavailable; callers can catch it
and fall back to the numpy implementation.

    from methods.common.cpp import CppInnerLoop, centroid_frame_cpp

    loop = CppInnerLoop(cfg, geom, ref_x, ref_y)   # precompute once
    cx, cy, slopes = loop.process(frame)           # per frame, in C++
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "centroid.cpp")
_LIB = os.path.join(_HERE, "libcentroid" + (".dylib" if sys.platform == "darwin" else ".so"))


class CppUnavailable(RuntimeError):
    """Raised when the C++ inner loop cannot be built or loaded."""


def _build() -> None:
    cc = os.environ.get("CXX") or ("c++" if sys.platform == "darwin" else "g++")
    flags = ["-O3", "-shared", "-fPIC", "-std=c++17"]
    cmd = [cc, *flags, _SRC, "-o", _LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CppUnavailable(f"compile failed ({' '.join(cmd)}):\n{proc.stderr}")


def _needs_build() -> bool:
    return (not os.path.exists(_LIB)) or (os.path.getmtime(_SRC) > os.path.getmtime(_LIB))


_lib = None


def _load():
    """Load (building if necessary) the shared library, once per process."""
    global _lib
    if _lib is not None:
        return _lib
    try:
        if _needs_build():
            _build()
        lib = ctypes.CDLL(_LIB)
    except CppUnavailable:
        raise
    except Exception as e:  # missing compiler, load error, ...
        raise CppUnavailable(f"could not load C++ inner loop: {e}") from e

    d_p = ctypes.POINTER(ctypes.c_double)
    i_p = ctypes.POINTER(ctypes.c_int)
    lib.centroid_slopes.restype = None
    lib.centroid_slopes.argtypes = [
        d_p, ctypes.c_int, ctypes.c_int,      # image, n, ppl
        i_p, i_p, ctypes.c_int,               # valid_i, valid_j, n_valid
        d_p, d_p,                             # ref_x, ref_y
        ctypes.c_double, ctypes.c_double,     # threshold_frac, slope_scale
        d_p, d_p, d_p,                        # cx_out, cy_out, slopes_out
    ]
    _lib = lib
    return lib


def is_available() -> bool:
    """True if the C++ inner loop can be built/loaded on this machine."""
    try:
        _load()
        return True
    except CppUnavailable:
        return False


def _dp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _ip(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


class CppInnerLoop:
    """Reusable C++ centroid+slope engine (precomputes geometry + buffers once).

    Buffers are allocated once and reused across frames, so per-call overhead is
    just the ctypes trampoline plus the C++ loop -- appropriate for a real-time
    control loop that processes one arriving frame at a time.
    """

    def __init__(self, cfg, geom, ref_x, ref_y, threshold_frac: float = 0.10):
        self.lib = _load()
        self.n = int(cfg.npix)
        self.ppl = int(cfg.pix_per_lenslet)
        self.nv = int(geom.n_valid)
        self.slope_scale = float(cfg.slope_scale)
        self.threshold_frac = float(threshold_frac)
        # Contiguous, dtype-locked geometry + reference arrays.
        self.vi = np.ascontiguousarray(geom.valid_idx[:, 0], dtype=np.int32)
        self.vj = np.ascontiguousarray(geom.valid_idx[:, 1], dtype=np.int32)
        self.ref_x = np.ascontiguousarray(ref_x, dtype=np.float64)
        self.ref_y = np.ascontiguousarray(ref_y, dtype=np.float64)
        # Output buffers (reused).
        self.cx = np.empty(self.nv, dtype=np.float64)
        self.cy = np.empty(self.nv, dtype=np.float64)
        self.slopes = np.empty(2 * self.nv, dtype=np.float64)

    def process(self, image: np.ndarray):
        """Frame -> (cx, cy, slopes). Returns fresh copies of the buffers."""
        img = np.ascontiguousarray(image, dtype=np.float64)
        if img.shape != (self.n, self.n):
            raise ValueError(f"expected ({self.n},{self.n}) frame, got {img.shape}")
        self.lib.centroid_slopes(
            _dp(img), self.n, self.ppl,
            _ip(self.vi), _ip(self.vj), self.nv,
            _dp(self.ref_x), _dp(self.ref_y),
            self.threshold_frac, self.slope_scale,
            _dp(self.cx), _dp(self.cy), _dp(self.slopes))
        return self.cx.copy(), self.cy.copy(), self.slopes.copy()

    def centroids(self, image: np.ndarray):
        """Frame -> absolute (cx, cy) centroids (matches numpy centroid_frame)."""
        cx, cy, _ = self.process(image)
        return cx, cy


def centroid_frame_cpp(image, cfg, geom, threshold_frac: float = 0.10):
    """Convenience: absolute (cx, cy) via C++, matching numpy centroid_frame.

    Builds a throwaway engine (with a zero reference, since centroids are
    reference-independent); for repeated calls construct a CppInnerLoop and
    reuse it instead.
    """
    zero = np.zeros(geom.n_valid, dtype=np.float64)
    loop = CppInnerLoop(cfg, geom, zero, zero, threshold_frac)
    return loop.centroids(image)
