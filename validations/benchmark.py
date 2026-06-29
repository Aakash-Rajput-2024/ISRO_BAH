"""Benchmark reconstruction methods head-to-head on a labelled dataset.

Scores every method on identical frames (with ground-truth wavefronts) and
prints/saves a scorecard: wavefront R^2, residual RMS, Strehl, r0 error, and
speed (ms/frame). This is how Method 1 (modal), Method 3 (ResU-Net), and any
future method are compared on equal footing.

    # Method 1 only (no training needed):
    python validations/benchmark.py --data data/synthetic/<name>
    # add Method 3 (needs a trained checkpoint):
    python validations/benchmark.py --data data/synthetic/<name> \
        --resunet methods/deep_resunet/weights/<name>.pt
    ./run.sh benchmark --data data/synthetic/<name> --resunet <ckpt.pt>

Any method exposing `.process(frame) -> obj` with `.wavefront` and `.coeffs`
(both Method 1's WFSPipeline and Method 3's ResUNetReconstructor do) can be
scored -- add it to `build_methods`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame
from methods.common.turbulence import estimate_r0
from validations import metrics as m

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def build_methods(cfg, resunet_ckpt=None):
    """Return {name: method} ready to `.process(frame)`."""
    methods = {}

    # Method 1 -- modal Zernike (calibrated on a noiseless flat frame).
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, pipe.geom))
    methods["modal_zernike"] = pipe

    # Method 3 -- ResU-Net (only if a trained checkpoint is supplied).
    if resunet_ckpt:
        from methods.deep_resunet import ResUNetReconstructor
        methods["deep_resunet"] = ResUNetReconstructor(resunet_ckpt, cfg)

    return methods


def score_method(method, frames, wavefronts, mask, true_r0, pupil_diameter):
    """Run a method over the dataset and return a scorecard dict."""
    n, n_modes = frames.shape[0], None
    r2s, rms = [], []
    coeffs = []
    t0 = time.perf_counter()
    for k in range(n):
        out = method.process(frames[k])
        recon = out.wavefront
        r2s.append(m.wavefront_r2(wavefronts[k], recon, mask))
        rms.append(m.residual_rms(wavefronts[k], recon, mask))
        coeffs.append(out.coeffs)
    ms_per_frame = 1e3 * (time.perf_counter() - t0) / n

    coeffs = np.asarray(coeffs)
    try:
        r0_est = estimate_r0(coeffs, pupil_diameter)
        r0_err = abs(r0_est - true_r0) / true_r0 if true_r0 else float("nan")
    except Exception:
        r0_est, r0_err = float("nan"), float("nan")

    mean_rms = float(np.mean(rms))
    return {
        "wavefront_R2": float(np.mean(r2s)),
        "residual_RMS_rad": mean_rms,
        "Strehl": m.strehl_estimate(mean_rms),
        "r0_rel_err": float(r0_err),
        "ms_per_frame": float(ms_per_frame),
    }


def run(data_path, resunet_ckpt=None):
    frames = np.load(os.path.join(data_path, "frames.npy")).astype(float)
    wavefronts = np.load(os.path.join(data_path, "wavefronts.npy")).astype(float)
    with open(os.path.join(data_path, "manifest.json")) as f:
        manifest = json.load(f)
    true_r0 = manifest.get("true_r0")

    cfg = Config()
    mask = WFSPipeline(cfg).geom.pupil_mask
    methods = build_methods(cfg, resunet_ckpt)

    scores = {}
    for name, method in methods.items():
        print(f"scoring {name} on {frames.shape[0]} frames ...")
        scores[name] = score_method(method, frames, wavefronts, mask, true_r0,
                                    cfg.pupil_diameter)

    _print_table(scores)
    _save(scores, data_path, manifest)
    return scores


_COLS = [
    ("wavefront_R2", "wavefront R²", "{:+.3f}", True),
    ("residual_RMS_rad", "resid RMS [rad]", "{:.3f}", False),
    ("Strehl", "Strehl", "{:.3f}", True),
    ("r0_rel_err", "r0 err", "{:.2%}", False),
    ("ms_per_frame", "ms/frame", "{:.2f}", False),
]


def _print_table(scores):
    names = list(scores)
    w = max(len(n) for n in names) + 2
    header = "method".ljust(w) + "".join(label.rjust(18) for _, label, _, _ in _COLS)
    print("\n" + header)
    print("-" * len(header))
    for n in names:
        row = n.ljust(w)
        for key, _, fmt, _ in _COLS:
            row += fmt.format(scores[n][key]).rjust(18)
        print(row)
    # Rank by wavefront R² (the headline metric).
    best = max(names, key=lambda n: scores[n]["wavefront_R2"])
    print(f"\nbest wavefront R²: {best}")


def _save(scores, data_path, manifest):
    os.makedirs(OUT, exist_ok=True)
    payload = {"dataset": os.path.basename(os.path.normpath(data_path)),
               "manifest": manifest, "scores": scores}
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = ["# Method comparison\n",
             f"Dataset: `{payload['dataset']}` "
             f"(D/r0={manifest.get('d_over_r0')}, n={manifest.get('n')})\n",
             "| method | " + " | ".join(l for _, l, _, _ in _COLS) + " |",
             "|" + "---|" * (len(_COLS) + 1)]
    for n, s in scores.items():
        lines.append("| " + n + " | " +
                      " | ".join(fmt.format(s[k]) for k, _, fmt, _ in _COLS) + " |")
    with open(os.path.join(OUT, "benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved -> {os.path.join(OUT, 'benchmark.md')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="data/synthetic/<name> folder")
    p.add_argument("--resunet", default=None, help="Method-3 checkpoint (.pt)")
    args = p.parse_args()
    run(args.data, resunet_ckpt=args.resunet)


if __name__ == "__main__":
    main()
