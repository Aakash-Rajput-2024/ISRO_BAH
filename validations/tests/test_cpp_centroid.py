"""Golden conformance: the C++ inner loop must match the numpy reference.

The C++ centroid+slope port (methods/common/cpp) is only trustworthy if it
reproduces methods/common/centroid.py bit-for-(almost)-bit. These tests render
realistic, noisy, 8-bit SH-WFS frames and assert the C++ centroids and slopes
agree with numpy to floating-point round-off -- the same oracle discipline the
suite already applies to the eventual C reconstruction port.

Skips cleanly (not fails) when no C++ compiler is available, so the suite still
runs on machines without a toolchain.
"""
import numpy as np
import pytest

from methods.common.centroid import centroid_frame
from methods.common.config import Config
from methods.common.geometry import Geometry
from methods.common.phasescreen import kolmogorov_screen
from methods.common.simulate import flat_frame, render_frame

cpp = pytest.importorskip("methods.common.cpp", reason="C++ inner loop module missing")

pytestmark = pytest.mark.golden

if not cpp.is_available():
    pytest.skip("no C++ compiler available to build the inner loop",
                allow_module_level=True)


def _realistic_frame(cfg, geom, rng, d_over_r0=6.0):
    """A noisy, 8-bit-quantised SH-WFS frame (matches the training realism)."""
    phase = kolmogorov_screen(cfg.npix, cfg.pupil_dx, cfg.pupil_diameter / d_over_r0, rng)
    phase = (phase - phase[geom.pupil_mask].mean()) * geom.pupil_mask
    return render_frame(phase, cfg, geom, peak=1500.0, read_noise=3.0,
                        photon_noise=True, bias=100.0, full_well=2000.0,
                        quantize_bits=8, rng=rng).astype(np.float64)


@pytest.fixture(scope="module")
def geom(cfg):
    return Geometry(cfg)


@pytest.mark.fast
def test_centroids_match_numpy(cfg, geom, rng):
    """Absolute centroids from C++ == numpy centroid_frame (to round-off)."""
    for _ in range(8):
        frame = _realistic_frame(cfg, geom, rng)
        cx_np, cy_np = centroid_frame(frame, cfg, geom)
        cx_c, cy_c = cpp.centroid_frame_cpp(frame, cfg, geom)
        # Sub-milli-pixel is already far tighter than any centroiding error;
        # float64 round-off puts this near 1e-12 px.
        assert np.allclose(cx_c, cx_np, atol=1e-9, rtol=0)
        assert np.allclose(cy_c, cy_np, atol=1e-9, rtol=0)


@pytest.mark.fast
def test_slopes_match_pipeline(cfg, geom, rng):
    """C++ slopes == (centroid - reference) * slope_scale from the numpy path."""
    ref_frame = flat_frame(cfg, geom).astype(np.float64)
    ref_x, ref_y = centroid_frame(ref_frame, cfg, geom)
    loop = cpp.CppInnerLoop(cfg, geom, ref_x, ref_y)

    for _ in range(8):
        frame = _realistic_frame(cfg, geom, rng)
        cx_np, cy_np = centroid_frame(frame, cfg, geom)
        slopes_np = np.concatenate([(cx_np - ref_x) * cfg.slope_scale,
                                    (cy_np - ref_y) * cfg.slope_scale])
        _, _, slopes_c = loop.process(frame)
        # Relative tolerance: slope magnitudes are ~1e4 rad/m, so scale the atol.
        scale = max(1.0, float(np.abs(slopes_np).max()))
        assert np.allclose(slopes_c, slopes_np, atol=1e-6 * scale, rtol=0)


@pytest.mark.fast
def test_empty_window_falls_back_to_centre(cfg, geom):
    """A zero (no-signal) frame -> every centroid sits at its window centre."""
    zero = np.zeros((cfg.npix, cfg.npix), dtype=np.float64)
    cx_np, cy_np = centroid_frame(zero, cfg, geom)
    cx_c, cy_c = cpp.centroid_frame_cpp(zero, cfg, geom)
    assert np.allclose(cx_c, cx_np, atol=1e-12, rtol=0)
    assert np.allclose(cy_c, cy_np, atol=1e-12, rtol=0)


@pytest.mark.fast
def test_shape_guard(cfg, geom):
    """A wrong-sized frame is rejected rather than reading out of bounds."""
    loop = cpp.CppInnerLoop(cfg, geom, np.zeros(geom.n_valid), np.zeros(geom.n_valid))
    with pytest.raises(ValueError):
        loop.process(np.zeros((cfg.npix // 2, cfg.npix), dtype=np.float64))
