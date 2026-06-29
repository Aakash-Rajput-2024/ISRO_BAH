"""Generate a labelled synthetic SH-WFS dataset into data/synthetic/.

Each dataset bundles detector frames with the ground truth that produced them
(input wavefronts, Zernike coefficients, true r0), so any method in `methods/`
can be trained/validated against an exact answer. This is the controllable
stand-in for the real lab .bmp series; raise the fidelity here (noise, outer
scale, subharmonics) as those generator upgrades land.

    python data/make_dataset.py                 # default ensemble dataset
    python data/make_dataset.py --kind frozen    # frozen-flow time series
    ./run.sh dataset
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from methods.modal_zernike import Config, WFSPipeline
from methods.common.phasescreen import frozen_flow_sequence, kolmogorov_screen
from methods.common.simulate import flat_frame, render_frame

OUT_ROOT = os.path.join(_ROOT, "data", "synthetic")


def _save_dataset(name, frames, wavefronts, coeffs, cfg, meta):
    out = os.path.join(OUT_ROOT, name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "frames.npy"), frames.astype(np.float32))
    np.save(os.path.join(out, "wavefronts.npy"), wavefronts.astype(np.float32))
    np.save(os.path.join(out, "coeffs.npy"), coeffs.astype(np.float32))
    manifest = {
        "n_frames": int(frames.shape[0]),
        "frame_shape": list(frames.shape[1:]),
        "config": {k: getattr(cfg, k) for k in
                   ("n_lenslets", "lenslet_pitch", "focal_length",
                    "pix_per_lenslet", "pixel_size", "wavelength", "n_modes")},
        "pupil_diameter": cfg.pupil_diameter,
        **meta,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {frames.shape[0]} frames -> {out}")


def make_ensemble(cfg, pipe, n, d_over_r0, read_noise, photon_noise, seed):
    rng = np.random.default_rng(seed)
    mask = pipe.geom.pupil_mask
    r0 = cfg.pupil_diameter / d_over_r0
    frames = np.empty((n, cfg.npix, cfg.npix))
    wfs = np.empty((n, cfg.npix, cfg.npix))
    coeffs = np.empty((n, cfg.n_modes))
    for k in range(n):
        ph = kolmogorov_screen(cfg.npix, cfg.pupil_dx, r0, rng)
        ph = (ph - ph[mask].mean()) * mask
        frame = render_frame(ph, cfg, pipe.geom, read_noise=read_noise,
                             photon_noise=photon_noise, rng=rng)
        frames[k], wfs[k] = frame, ph
        coeffs[k] = pipe.process(frame).coeffs
    meta = {"kind": "ensemble", "d_over_r0": d_over_r0, "true_r0": r0,
            "read_noise": read_noise, "photon_noise": photon_noise, "seed": seed}
    return frames, wfs, coeffs, meta


def make_frozen(cfg, pipe, n, d_over_r0, wind, seed):
    rng = np.random.default_rng(seed)
    mask = pipe.geom.pupil_mask
    r0 = cfg.pupil_diameter / d_over_r0
    dt = cfg.pupil_dx / wind
    seq = list(frozen_flow_sequence(cfg.npix, cfg.pupil_dx, r0, n, wind, dt, rng))
    frames = np.empty((n, cfg.npix, cfg.npix))
    wfs = np.empty((n, cfg.npix, cfg.npix))
    coeffs = np.empty((n, cfg.n_modes))
    for k, ph in enumerate(seq):
        ph = (ph - ph[mask].mean()) * mask
        frame = render_frame(ph, cfg, pipe.geom, rng=rng)
        frames[k], wfs[k] = frame, ph
        coeffs[k] = pipe.process(frame).coeffs
    meta = {"kind": "frozen", "d_over_r0": d_over_r0, "true_r0": r0,
            "wind": wind, "dt": dt, "seed": seed}
    return frames, wfs, coeffs, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["ensemble", "frozen"], default="ensemble")
    p.add_argument("--name", default=None, help="output subdir under data/synthetic/")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--d-over-r0", type=float, default=6.0)
    p.add_argument("--wind", type=float, default=5.0, help="frozen-flow wind [m/s]")
    p.add_argument("--read-noise", type=float, default=2.0)
    p.add_argument("--no-photon-noise", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, pipe.geom))

    if args.kind == "ensemble":
        data = make_ensemble(cfg, pipe, args.n, args.d_over_r0,
                             args.read_noise, not args.no_photon_noise, args.seed)
        name = args.name or f"ensemble_dr0_{args.d_over_r0:g}_n{args.n}"
    else:
        data = make_frozen(cfg, pipe, args.n, args.d_over_r0, args.wind, args.seed)
        name = args.name or f"frozen_v{args.wind:g}_n{args.n}"

    frames, wfs, coeffs, meta = data
    _save_dataset(name, frames, wfs, coeffs, cfg, meta)


if __name__ == "__main__":
    main()
