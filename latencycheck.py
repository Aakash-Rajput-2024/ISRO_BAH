#!/usr/bin/env python3
"""End-to-end latency benchmark: raw SH-WFS frame -> wavefront, per method.

Answers the real-time question the accuracy-focused tools (validations/
benchmark.py, methods/hrnet/eval.py) don't: for a single frame arriving off
the detector, how long does the FULL reconstruction chain take before a
wavefront / actuator command is available. Not just the inner matrix-vector
product -- centroiding, preprocessing, model inference, coefficient
extraction, exactly as it would run in the control loop.

    modal_zernike       frame -> centroid -> slopes -> pinv(D)@s -> W   (numpy)
    modal_zernike_cpp   same, but centroid+slopes run in compiled C++ (the
                        ~1.9 ms hot spot; see methods/common/cpp/)
    deep_resunet        frame -> average-added -> CNN -> upsample -> W
    hrnet_full          [frame_t, frame_t-1, ...] -> CNN -> W(t)  (18ch baseline)
    hrnet_small         depth-trimmed HRNet retrained to fit the 10 ms budget

Frames are realistic, not random noise: multi-layer frozen-flow + boiling
turbulence rendered through the same Gaussian-spot + photon/read-noise +
8-bit detector model used to generate the training data (see
methods/common/simulate.py, methods/common/phasescreen.py, and
data/make_sequence_dataset.py, whose detector/turbulence defaults this
script reuses verbatim). Every method sees the identical frame content at
each iteration, so the comparison is apples-to-apples.

Every method is timed per-frame (one call per arriving frame, not amortised
over a batch -- that's how a control loop actually sees it), with warm-up
calls discarded so first-call overhead (cuDNN/MPS kernel build, buffer
allocation) doesn't leak into the numbers, and the device is synchronised
before the clock stops so a queued GPU/MPS launch is actually finished, not
just submitted. Reports mean/median/p95/p99/max plus achievable frame rate:
tail latency is what blows a real-time deadline, not the mean.

    python latencycheck.py                       # all methods, auto device
    python latencycheck.py --iters 500 --warmup 50
    python latencycheck.py --device cpu           # force CPU-only comparison
    python latencycheck.py --skip-hrnet --skip-resunet
    ./run.sh latency --iters 500

Methods are skipped with a warning (not an error) when their trained
checkpoint is absent, or -- for modal_zernike_cpp -- when no C++ compiler is
available to build the inner loop.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from methods.common.config import Config
from methods.common.geometry import Geometry
from methods.common.phasescreen import multilayer_boiling_sequence
from methods.common.simulate import render_frame, flat_frame
from methods.modal_zernike import WFSPipeline

try:
    import torch
except ImportError:
    torch = None

OUT = os.path.join(_ROOT, "outputs")
DEFAULT_RESUNET_CKPT = os.path.join(_ROOT, "methods", "deep_resunet", "weights", "train_dr0_6.pt")
DEFAULT_HRNET_CKPT = os.path.join(_ROOT, "methods", "hrnet", "weights", "seq_train.pt")
DEFAULT_HRNET_SMALL_CKPT = os.path.join(_ROOT, "methods", "hrnet", "weights", "seq_train_small.pt")

# Detector + turbulence realism, copied from data/make_sequence_dataset.py's
# defaults so these frames match what the checkpoints were trained/scored on.
DETECTOR = dict(peak=1500.0, read_noise=3.0, bias=100.0, full_well=2000.0,
                 quantize_bits=8, photon_noise=True)
TURBULENCE = dict(n_layers=2, d_over_r0=(4.0, 10.0), wind=(3.0, 12.0), L0=0.1,
                   boil_frac=0.15, boil_alpha=0.1, n_subharmonics=3,
                   shift_px=0.5, v_ref=7.0)


# --------------------------------------------------------------------------- #
# Realistic frame stream (shared, unmodified, across every method).
# --------------------------------------------------------------------------- #
def make_frame_stream(cfg: Config, geom: Geometry, n_frames: int,
                       rng: np.random.Generator):
    """One continuous, realistically-turbulent SH-WFS frame sequence."""
    dr0 = rng.uniform(*TURBULENCE["d_over_r0"])
    r0 = cfg.pupil_diameter / dr0
    layers = [(rng.uniform(0.3, 1.0), rng.uniform(*TURBULENCE["wind"]),
               rng.uniform(0.0, 2 * np.pi)) for _ in range(TURBULENCE["n_layers"])]
    dt = TURBULENCE["shift_px"] * cfg.pupil_dx / TURBULENCE["v_ref"]
    seq = multilayer_boiling_sequence(
        cfg.npix, cfg.pupil_dx, n_frames, r0, layers, dt, rng=rng,
        boil_frac=TURBULENCE["boil_frac"], boil_alpha=TURBULENCE["boil_alpha"],
        L0=TURBULENCE["L0"], n_subharmonics=TURBULENCE["n_subharmonics"])
    mask = geom.pupil_mask
    frames = []
    for phase in seq:
        phase = (phase - phase[mask].mean()) * mask
        fr = render_frame(phase, cfg, geom, rng=rng, **DETECTOR)
        frames.append(fr.astype(np.float32))
    meta = {"d_over_r0": float(dr0), "true_r0_m": float(r0),
            "wind_speeds_mps": [float(l[1]) for l in layers]}
    return frames, meta


def windows(frames, n_hist):
    """Sliding history windows, newest-first (matches HRNet's channel order)."""
    for i in range(n_hist, len(frames)):
        hist = frames[i - n_hist: i + 1][::-1]
        yield np.stack(hist, axis=0)


# --------------------------------------------------------------------------- #
# Timing primitives.
# --------------------------------------------------------------------------- #
def _sync(device) -> None:
    if torch is None or device is None:
        return
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_calls(fn, n_warmup: int, n_iters: int, device=None) -> np.ndarray:
    """Per-call latency [ms] for `n_iters` calls, after `n_warmup` discarded."""
    for _ in range(n_warmup):
        fn()
    _sync(device)
    out = np.empty(n_iters)
    for k in range(n_iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out[k] = (time.perf_counter() - t0) * 1e3
    return out


def stats(ms: np.ndarray) -> dict:
    mean = float(np.mean(ms))
    return {
        "n": int(ms.size),
        "mean_ms": mean,
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(np.min(ms)),
        "max_ms": float(np.max(ms)),
        "std_ms": float(np.std(ms)),
        "fps": float(1000.0 / mean) if mean > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Per-method benchmarks. Each times the SAME `current_frames` slice, one
# frame per call -- the real deployment shape.
# --------------------------------------------------------------------------- #
def bench_modal(cfg, geom, current_frames, warmup, iters):
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, geom))
    it = iter(current_frames)

    def step():
        pipe.process(next(it))

    ms = time_calls(step, warmup, iters)

    # Stage breakdown (centroid vs. modal reconstruction) on a fresh sample --
    # the README calls this pairing out explicitly as the piece that must hit
    # ~10 ms/frame in the eventual C port.
    sample = current_frames[-min(50, len(current_frames)):]
    c_ms, r_ms = [], []
    for fr in sample:
        t0 = time.perf_counter()
        _, _, slopes = pipe.slopes_from_frame(fr)
        t1 = time.perf_counter()
        coeffs = pipe.recon.coeffs_from_slopes(slopes)
        pipe.recon.wavefront_from_coeffs(coeffs)
        t2 = time.perf_counter()
        c_ms.append((t1 - t0) * 1e3)
        r_ms.append((t2 - t1) * 1e3)

    breakdown = {"centroid": float(np.mean(c_ms)), "reconstruct": float(np.mean(r_ms))}
    return {"stats": stats(ms), "device": "cpu (numpy)", "breakdown": breakdown}


def bench_modal_cpp(cfg, geom, current_frames, warmup, iters):
    """Modal method with the centroid+slope inner loop in C++ (numpy matmul kept).

    Only the per-sub-aperture pixel loop (the ~1.9 ms hot spot) moves to C++;
    the reconstruction stays a numpy/BLAS matrix-vector product. This is the
    README's 'C port of the centroiding inner loop'.
    """
    from methods.common.cpp import CppInnerLoop
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, geom))                # numpy reference (one-time)
    loop = CppInnerLoop(cfg, geom, pipe.ref_x, pipe.ref_y)
    recon = pipe.recon
    it = iter(current_frames)

    def step():
        _, _, slopes = loop.process(next(it))
        coeffs = recon.coeffs_from_slopes(slopes)
        recon.wavefront_from_coeffs(coeffs)

    ms = time_calls(step, warmup, iters)

    sample = current_frames[-min(50, len(current_frames)):]
    c_ms, r_ms = [], []
    for fr in sample:
        t0 = time.perf_counter()
        _, _, slopes = loop.process(fr)
        t1 = time.perf_counter()
        coeffs = recon.coeffs_from_slopes(slopes)
        recon.wavefront_from_coeffs(coeffs)
        t2 = time.perf_counter()
        c_ms.append((t1 - t0) * 1e3)
        r_ms.append((t2 - t1) * 1e3)

    breakdown = {"cpp_inner_loop": float(np.mean(c_ms)), "reconstruct": float(np.mean(r_ms))}
    return {"stats": stats(ms), "device": "cpu (C++ inner loop + numpy matmul)",
            "breakdown": breakdown}


def bench_resunet(cfg, ckpt_path, current_frames, warmup, iters, device):
    from methods.deep_resunet import ResUNetReconstructor
    recon = ResUNetReconstructor(ckpt_path, cfg, device=device)
    it = iter(current_frames)

    def step():
        recon.process(next(it))

    ms = time_calls(step, warmup, iters, device=device)
    return {"stats": stats(ms), "device": str(device), "checkpoint": os.path.basename(ckpt_path)}


def bench_hrnet(cfg, ckpt, frames, warmup, iters, device):
    from methods.hrnet import HRNetWavefront
    n_frames, w = ckpt["n_frames"], ckpt["out_size"]
    # Depth knobs default to the original architecture for pre-existing checkpoints.
    model = HRNetWavefront(n_frames=n_frames, base_channels=ckpt["channels"],
                           blocks_per_branch=ckpt.get("blocks_per_branch", 2),
                           frame_depth_growth=ckpt.get("frame_depth_growth", 1),
                           head="map", out_size=w).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_hist = n_frames - 1
    win_it = windows(frames, n_hist)

    def step():
        x = next(win_it) / 255.0
        xt = torch.from_numpy(x)[None].float().to(device)
        with torch.no_grad():
            model(xt)

    ms = time_calls(step, warmup, iters, device=device)
    return {"stats": stats(ms), "device": str(device), "n_frames_history": n_frames,
            "channels": ckpt["channels"], "n_params": int(n_params),
            "blocks_per_branch": ckpt.get("blocks_per_branch", 2),
            "frame_depth_growth": ckpt.get("frame_depth_growth", 1)}


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
_COLS = [
    ("mean_ms", "mean[ms]", "{:.3f}"),
    ("median_ms", "median[ms]", "{:.3f}"),
    ("p95_ms", "p95[ms]", "{:.3f}"),
    ("p99_ms", "p99[ms]", "{:.3f}"),
    ("max_ms", "max[ms]", "{:.3f}"),
    ("fps", "fps", "{:.1f}"),
]


def print_report(scores: dict, meta: dict, budget_ms: float) -> None:
    print("\n--- environment ---")
    for k, v in meta["environment"].items():
        print(f"  {k}: {v}")
    print("\n--- input (realistic, shared across all methods) ---")
    for k, v in meta["frame_stream"].items():
        print(f"  {k}: {v}")

    names = list(scores)
    w = max(len(n) for n in names) + 2
    header = "method".ljust(w) + "".join(l.rjust(13) for _, l, _ in _COLS) + "budget(p99)".rjust(14)
    print("\n" + header)
    print("-" * len(header))
    for n in names:
        s = scores[n]["stats"]
        row = n.ljust(w)
        for key, _, fmt in _COLS:
            row += fmt.format(s[key]).rjust(13)
        verdict = "PASS" if s["p99_ms"] <= budget_ms else "FAIL"
        row += verdict.rjust(14)
        print(row)

    for n in names:
        b = scores[n].get("breakdown")
        if b:
            parts = " + ".join(f"{stage} {ms:.3f} ms" for stage, ms in b.items())
            print(f"\n{n} stage breakdown: {parts}")

    print(f"\nbudget: {budget_ms:.1f} ms/frame (p99) -- "
          "README target for the eventual C-ported inner loop")
    fastest = min(scores, key=lambda n: scores[n]["stats"]["mean_ms"])
    print(f"fastest (mean): {fastest}")


def save_report(scores: dict, meta: dict, budget_ms: float) -> None:
    os.makedirs(OUT, exist_ok=True)
    payload = {"meta": meta, "budget_ms": budget_ms, "scores": scores}
    with open(os.path.join(OUT, "latency.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = ["# End-to-end latency\n",
             f"Device resolution: `{meta['environment'].get('resolved_device')}`  "
             f"·  frame `{meta['frame_stream']['npix']}x{meta['frame_stream']['npix']}`  "
             f"·  D/r0={meta['frame_stream']['d_over_r0']:.2f}\n",
             "| method | " + " | ".join(l for _, l, _ in _COLS) + " | budget(p99) |",
             "|" + "---|" * (len(_COLS) + 2)]
    for n, s in scores.items():
        st = s["stats"]
        verdict = "PASS" if st["p99_ms"] <= budget_ms else "FAIL"
        lines.append("| " + n + " | " +
                      " | ".join(fmt.format(st[k]) for k, _, fmt in _COLS) +
                      f" | {verdict} |")
    with open(os.path.join(OUT, "latency.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved -> {os.path.join(OUT, 'latency.json')}, {os.path.join(OUT, 'latency.md')}")


# --------------------------------------------------------------------------- #
def resolve_device(name: str):
    if torch is None:
        return None
    if name == "auto":
        from methods.deep_resunet import pick_device
        return pick_device()
    return torch.device(name)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=200, help="timed frames per method")
    p.add_argument("--warmup", type=int, default=20, help="discarded warm-up frames")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--budget-ms", type=float, default=10.0,
                   help="p99 latency budget [ms] (README target for the C port)")
    p.add_argument("--resunet-ckpt", default=None)
    p.add_argument("--hrnet-ckpt", default=None, help="full HRNet checkpoint")
    p.add_argument("--hrnet-small-ckpt", default=None,
                   help="depth-trimmed HRNet checkpoint (defaults to seq_train_small.pt)")
    p.add_argument("--skip-modal", action="store_true")
    p.add_argument("--skip-modal-cpp", action="store_true", help="skip the C++ modal path")
    p.add_argument("--skip-resunet", action="store_true")
    p.add_argument("--skip-hrnet", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    resunet_ckpt = args.resunet_ckpt or (
        DEFAULT_RESUNET_CKPT if os.path.exists(DEFAULT_RESUNET_CKPT) else None)

    # HRNet checkpoints to time: the full model and (if present) the small one.
    hrnet_specs = []          # (label, path)
    if not args.skip_hrnet and torch is not None:
        full = args.hrnet_ckpt or (DEFAULT_HRNET_CKPT if os.path.exists(DEFAULT_HRNET_CKPT) else None)
        small = args.hrnet_small_ckpt or (
            DEFAULT_HRNET_SMALL_CKPT if os.path.exists(DEFAULT_HRNET_SMALL_CKPT) else None)
        if full:
            hrnet_specs.append(("hrnet_full", full))
        if small:
            hrnet_specs.append(("hrnet_small", small))

    # C++ modal path is available only if the inner loop can be built/loaded.
    run_modal_cpp = False
    if not args.skip_modal_cpp:
        try:
            from methods.common.cpp import is_available
            run_modal_cpp = is_available()
            if not run_modal_cpp:
                print("skipping modal_zernike_cpp: no C++ compiler available")
        except Exception as e:
            print(f"skipping modal_zernike_cpp: {e}")

    run_modal = not args.skip_modal
    run_resunet = not args.skip_resunet and torch is not None and resunet_ckpt is not None

    if not args.skip_resunet and (torch is None or resunet_ckpt is None):
        reason = "torch not installed" if torch is None else f"no checkpoint at {DEFAULT_RESUNET_CKPT}"
        print(f"skipping deep_resunet: {reason}")
    if not args.skip_hrnet and torch is None:
        print("skipping hrnet: torch not installed")
    elif not args.skip_hrnet and not hrnet_specs:
        print(f"skipping hrnet: no checkpoint (looked for {DEFAULT_HRNET_CKPT})")
    if not (run_modal or run_modal_cpp or run_resunet or hrnet_specs):
        print("nothing to benchmark -- all methods skipped"); return

    cfg = Config()
    geom = Geometry(cfg)
    rng = np.random.default_rng(args.seed)

    # Load HRNet checkpoints; frame history must cover the deepest model.
    hrnet_ckpts = [(label, torch.load(path, map_location="cpu", weights_only=False))
                   for label, path in hrnet_specs]
    n_hist = max([ck["n_frames"] - 1 for _, ck in hrnet_ckpts], default=0)

    n_total = args.warmup + args.iters + n_hist
    print(f"generating {n_total} realistic SH-WFS frames "
          f"({cfg.npix}x{cfg.npix}, {geom.n_valid} valid sub-apertures) ...")
    frames, stream_meta = make_frame_stream(cfg, geom, n_total, rng)
    current_frames = frames[n_hist:]

    scores = {}
    if run_modal:
        print("timing modal_zernike (numpy) ...")
        scores["modal_zernike"] = bench_modal(cfg, geom, current_frames, args.warmup, args.iters)
    if run_modal_cpp:
        print("timing modal_zernike_cpp (C++ inner loop) ...")
        scores["modal_zernike_cpp"] = bench_modal_cpp(cfg, geom, current_frames,
                                                      args.warmup, args.iters)
    if run_resunet:
        print("timing deep_resunet ...")
        scores["deep_resunet"] = bench_resunet(cfg, resunet_ckpt, current_frames,
                                               args.warmup, args.iters, device)
    for label, ck in hrnet_ckpts:
        blk, grow = ck.get("blocks_per_branch", 2), ck.get("frame_depth_growth", 1)
        print(f"timing {label} (n_frames={ck['n_frames']}, {ck['channels']}ch, "
              f"blk={blk}, grow={grow}) ...")
        scores[label] = bench_hrnet(cfg, ck, frames, args.warmup, args.iters, device)

    meta = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": (torch.__version__ if torch is not None else "not installed"),
            "resolved_device": str(device),
        },
        "frame_stream": {
            "npix": cfg.npix, "n_valid_subapertures": geom.n_valid, "n_modes": cfg.n_modes,
            "n_total_frames": n_total, "warmup": args.warmup, "iters": args.iters,
            "seed": args.seed, **stream_meta,
        },
    }
    print_report(scores, meta, args.budget_ms)
    if not args.no_save:
        save_report(scores, meta, args.budget_ms)


if __name__ == "__main__":
    main()
