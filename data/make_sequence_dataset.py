"""Generate a TEMPORAL SH-WFS dataset shaped (N, T, D, D) into data/synthetic/.

N independent turbulence instances, each a continuous sequence of T frames
(multi-layer frozen flow + boiling + von Karman + subharmonics), rendered with a
realistic 8-bit detector. Each timestep carries ground truth (true Zernike
coeffs, true slopes, downsampled true wavefront) so a temporal model that takes
frames [t-K+1 .. t] can be trained/validated against an exact answer.

    python data/make_sequence_dataset.py --n 200 --t 32 --name seq_v1
    ./run.sh seqdataset --n 200 --t 32 --name seq_v1

Until ISRO provides real .bmp frames, this is the realism stand-in: tune the
fidelity knobs (--bits, --read-noise, --boil-frac, --L0) so synthetic frames
behave like the lab camera.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from numpy.lib.format import open_memmap

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from methods.modal_zernike import Config
from methods.common.geometry import Geometry
from methods.common.zernike import zernike_basis
from methods.common.phasescreen import multilayer_boiling_sequence
from methods.common.simulate import render_frame, true_slopes

OUT_ROOT = os.path.join(_ROOT, "data", "synthetic")


def _projection(cfg, geom):
    """Zernike basis + masked projection operator for TRUE coeffs."""
    modes = list(range(2, cfg.n_modes + 2))
    zmaps = zernike_basis(modes, geom.rho, geom.theta, geom.pupil_mask)
    A = zmaps[:, geom.pupil_mask].T          # (n_pix, n_modes)
    pinvA = np.linalg.pinv(A)                # (n_modes, n_pix)
    return pinvA


def _downsample(phase, w):
    """Area-average a (D,D) map to (w,w); requires D % w == 0."""
    D = phase.shape[0]
    b = D // w
    return phase.reshape(w, b, w, b).mean(axis=(1, 3))


def generate(cfg, out_dir, n, t, *, n_layers=2, d_over_r0=(4.0, 10.0),
             wind=(3.0, 12.0), L0=0.1, boil_frac=0.15, boil_alpha=0.1,
             shift_px=0.5, v_ref=7.0, peak=1500.0, read_noise=3.0, bias=100.0,
             full_well=2000.0, bits=8, target_w=48, n_subharmonics=3, seed=0):
    geom = Geometry(cfg)
    mask = geom.pupil_mask
    pinvA = _projection(cfg, geom)
    D, M, V = cfg.npix, cfg.n_modes, 2 * geom.n_valid
    dt = shift_px * cfg.pupil_dx / v_ref           # one dataset-wide dt
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    frames = open_memmap(os.path.join(out_dir, "frames.npy"), mode="w+",
                         dtype=np.uint8 if bits <= 8 else np.float16,
                         shape=(n, t, D, D))
    wfs = open_memmap(os.path.join(out_dir, "wavefronts.npy"), mode="w+",
                      dtype=np.float16, shape=(n, t, target_w, target_w))
    coeffs = np.empty((n, t, M), np.float32)
    slopes = np.empty((n, t, V), np.float32)
    mean_frame = np.empty((n, D, D), np.float32)
    params = []

    for i in range(n):
        dr0 = rng.uniform(*d_over_r0)
        r0 = cfg.pupil_diameter / dr0
        layers = [(rng.uniform(0.3, 1.0), rng.uniform(*wind),
                   rng.uniform(0, 2 * np.pi)) for _ in range(n_layers)]
        seq = multilayer_boiling_sequence(
            D, cfg.pupil_dx, t, r0, layers, dt, rng=rng, boil_frac=boil_frac,
            boil_alpha=boil_alpha, L0=L0, n_subharmonics=n_subharmonics)
        inst_frames = np.empty((t, D, D), np.float32)
        for k, phase in enumerate(seq):
            phase = (phase - phase[mask].mean()) * mask
            fr = render_frame(phase, cfg, geom, peak=peak, read_noise=read_noise,
                              photon_noise=True, bias=bias, full_well=full_well,
                              quantize_bits=bits, rng=rng)
            inst_frames[k] = fr
            frames[i, k] = fr.astype(frames.dtype)
            coeffs[i, k] = pinvA @ phase[mask]
            slopes[i, k] = true_slopes(phase, cfg, geom)
            wfs[i, k] = _downsample(phase, target_w).astype(np.float16)
        mean_frame[i] = inst_frames.mean(axis=0)     # per-instance time-average
        params.append({"d_over_r0": dr0, "r0": r0,
                       "layers": [[float(c), float(s), float(d)] for c, s, d in layers]})
        print(f"  instance {i+1}/{n}  D/r0={dr0:.1f}  layers={n_layers}")

    frames.flush(); wfs.flush()
    np.save(os.path.join(out_dir, "coeffs.npy"), coeffs)
    np.save(os.path.join(out_dir, "slopes.npy"), slopes)
    np.save(os.path.join(out_dir, "mean_frame.npy"), mean_frame)

    # By-instance train/val/test split (no temporal leakage across the N axis).
    idx = np.random.default_rng(seed + 1).permutation(n).tolist()
    n_test = max(1, n // 5); n_val = max(1, n // 5)
    split = {"test": idx[:n_test], "val": idx[n_test:n_test + n_val],
             "train": idx[n_test + n_val:]}

    manifest = {
        "kind": "sequence", "N": n, "T": t, "D": D, "n_modes": M, "n_slopes": V,
        "target_w": target_w, "dt": dt, "frames_dtype": str(frames.dtype),
        "frame_units": "8-bit counts" if bits <= 8 else "float",
        "split": split,
        "generator": {"n_layers": n_layers, "d_over_r0": list(d_over_r0),
                      "wind": list(wind), "L0": L0, "boil_frac": boil_frac,
                      "boil_alpha": boil_alpha, "n_subharmonics": n_subharmonics,
                      "shift_px": shift_px, "v_ref": v_ref},
        "detector": {"peak": peak, "read_noise": read_noise, "bias": bias,
                     "full_well": full_well, "bits": bits},
        "config": {"n_lenslets": cfg.n_lenslets, "pix_per_lenslet": cfg.pix_per_lenslet,
                   "pupil_diameter": cfg.pupil_diameter},
        "params": params,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote (N={n}, T={t}, D={D}) -> {out_dir}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200, help="number of instances")
    p.add_argument("--t", type=int, default=32, help="frames per instance")
    p.add_argument("--name", default=None)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--L0", type=float, default=0.1, help="outer scale [m]")
    p.add_argument("--boil-frac", type=float, default=0.15)
    p.add_argument("--boil-alpha", type=float, default=0.1)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--read-noise", type=float, default=3.0)
    p.add_argument("--target-w", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = Config()
    name = args.name or f"seq_n{args.n}_t{args.t}"
    out = os.path.join(OUT_ROOT, name)
    generate(cfg, out, args.n, args.t, n_layers=args.layers, L0=args.L0,
             boil_frac=args.boil_frac, boil_alpha=args.boil_alpha, bits=args.bits,
             read_noise=args.read_noise, target_w=args.target_w, seed=args.seed)


if __name__ == "__main__":
    main()
