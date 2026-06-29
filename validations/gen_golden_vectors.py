"""Regenerate the golden oracle vectors for the C port (Tier 4).

Run deliberately and commit the result; `tests/test_golden_vectors.py` then
asserts the Python reference still reproduces them, and the C port will be
diffed against the same fixtures.

    python scripts/gen_golden_vectors.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame, render_frame
from validations import oracle


def main() -> None:
    cfg = Config()
    pipe = WFSPipeline(cfg)
    flat = flat_frame(cfg, pipe.geom)
    pipe.calibrate(flat)

    # A fixed, reproducible aberration (noiseless) so every stage is deterministic.
    coeffs_in = np.zeros(cfg.n_modes)
    for j, a in {3: 0.4, 5: 0.8, 8: -0.5, 11: 0.3, 14: 0.2}.items():
        coeffs_in[j - 2] = a
    phase = pipe.recon.wavefront_from_coeffs(coeffs_in)
    frame = render_frame(phase, cfg, pipe.geom)  # noiseless, deterministic

    # Run the inner loop, capturing the I/O of each stage.
    cx, cy, slopes = pipe.slopes_from_frame(frame)
    coeffs = pipe.recon.coeffs_from_slopes(slopes)
    wavefront = pipe.recon.wavefront_from_coeffs(coeffs)

    arrays = {
        # stage inputs
        "flat_frame": flat,
        "frame": frame,
        "ref_x": pipe.ref_x,
        "ref_y": pipe.ref_y,
        # stage outputs
        "cx": cx,
        "cy": cy,
        "slopes": slopes,
        "coeffs": coeffs,
        "wavefront": wavefront,
        # operators (store R explicitly; do not recompute pinv in C)
        "D": pipe.recon.D,
        "R": pipe.recon.R,
        # ground truth for reference
        "coeffs_in": coeffs_in,
    }
    oracle.dump_golden(arrays, cfg)
    print(f"wrote {len(arrays)} arrays to {oracle.GOLDEN_DIR}")
    print(f"config_hash = {oracle.config_hash(cfg)}")


if __name__ == "__main__":
    main()
