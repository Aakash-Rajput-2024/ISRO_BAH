"""Evaluate a trained HRNet temporal model (Method 4) -- the ML validation.

Runs on the dataset's HELD-OUT test instances (never in train/val, no temporal
leakage) and answers the two questions that matter for a learned model:
  1. does it generalise?            -> wavefront R^2, residual RMS, Strehl
  2. does it beat the baseline?     -> same metrics for Method 1 (modal) on the
                                       same frames, side by side
plus a physics-consistency check (does the predicted wavefront's RMS track the
true RMS?) and example truth/prediction/residual panels.

    python methods/hrnet/eval.py                       # default dataset + checkpoint
    python methods/hrnet/eval.py --data data/synthetic/seq_v2 --ckpt path.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from methods.common.sequence_dataset import load_sequences, split_from_manifest
from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame
from methods.hrnet import HRNetWavefront, pick_device
from validations import metrics as m

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def _down(a, w):
    """Area-average a (..., D, D) map to (..., w, w); D % w == 0."""
    D = a.shape[-1]
    b = D // w
    return a.reshape(*a.shape[:-2], w, b, w, b).mean(axis=(-3, -1))


def evaluate(data_path, ckpt_path, max_windows=300, n_panels=4):
    device = pick_device()
    data = load_sequences(data_path)
    test = split_from_manifest(data)["test"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_frames, w, wf_scale = ckpt["n_frames"], ckpt["out_size"], ckpt["wf_scale"]
    # Depth knobs default to the original architecture for pre-existing checkpoints.
    model = HRNetWavefront(n_frames=n_frames, base_channels=ckpt["channels"],
                           blocks_per_branch=ckpt.get("blocks_per_branch", 2),
                           frame_depth_growth=ckpt.get("frame_depth_growth", 1),
                           head="map", out_size=w).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {os.path.basename(ckpt_path)}  "
          f"({n_frames}f, {ckpt['channels']}ch, blk={ckpt.get('blocks_per_branch', 2)}, "
          f"grow={ckpt.get('frame_depth_growth', 1)}, {n_params/1e3:.0f}k params)")

    cfg = Config()
    pipe = WFSPipeline(cfg)
    pipe.calibrate(flat_frame(cfg, pipe.geom))           # Method 1 baseline
    cmask = _down(pipe.geom.pupil_mask.astype(float), w) > 0.5

    view = data.windows(n_frames, horizon=0, label="wavefronts", instances=test)
    n = min(len(view), max_windows)
    idxs = np.linspace(0, len(view) - 1, n).astype(int)

    rows = {"hrnet": {"r2": [], "rms": []}, "modal": {"r2": [], "rms": []}}
    true_rms, pred_rms = [], []
    panels = []
    for k, i in enumerate(idxs):
        x, truth = view[i]                               # x:(K,D,D)[0,1]; truth:(w,w)[rad]
        # --- Method 4 (HRNet, temporal) ---
        xin = torch.from_numpy(np.array(x[::-1]))[None].float().to(device)
        with torch.no_grad():
            pred = model(xin).cpu().numpy()[0, 0] * wf_scale
        rows["hrnet"]["r2"].append(m.wavefront_r2(truth, pred, cmask))
        rows["hrnet"]["rms"].append(m.residual_rms(truth, pred, cmask))
        true_rms.append(truth[cmask].std()); pred_rms.append(pred[cmask].std())
        # --- Method 1 (modal) on the SAME current frame t ---
        frame_t = x[-1].astype(float)                    # current frame (scale-invariant CoG)
        base = _down(pipe.process(frame_t).wavefront, w)
        rows["modal"]["r2"].append(m.wavefront_r2(truth, base, cmask))
        rows["modal"]["rms"].append(m.residual_rms(truth, base, cmask))
        if len(panels) < n_panels and k % max(1, n // n_panels) == 0:
            panels.append((truth, pred, base, cmask))

    def agg(d):
        r2, rms = np.array(d["r2"]), np.array(d["rms"])
        return r2.mean(), rms.mean(), m.strehl_estimate(rms.mean())

    report = {name: agg(d) for name, d in rows.items()}
    rms_corr = float(np.corrcoef(true_rms, pred_rms)[0, 1])
    _print(report, n, rms_corr)
    _panels(panels, os.path.join(OUT_DIR, "eval_panels.png"))
    return report


def _print(report, n, rms_corr):
    print(f"\nHeld-out test set ({n} windows)            wavefront R²   resid RMS [rad]   Strehl")
    print("-" * 76)
    for name in ("hrnet", "modal"):
        r2, rms, st = report[name]
        label = "Method 4 (HRNet temporal)" if name == "hrnet" else "Method 1 (modal)"
        print(f"  {label:30s} {r2:+13.3f} {rms:17.3f} {st:8.3f}")
    win = "HRNet" if report["hrnet"][0] > report["modal"][0] else "modal"
    print(f"\nbetter wavefront R²: {win}")
    print(f"physics check: corr(true RMS, predicted RMS) = {rms_corr:+.3f}  (→1 = tracks turbulence strength)")


def _panels(panels, path):
    if not panels:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(panels), 4, figsize=(11, 2.6 * len(panels)))
    ax = np.atleast_2d(ax)
    titles = ["truth W(t)", "HRNet", "modal", "HRNet residual"]
    for r, (truth, pred, base, mask) in enumerate(panels):
        vmax = np.nanmax(np.abs(np.where(mask, truth, np.nan)))
        imgs = [np.where(mask, a, np.nan) for a in (truth, pred, base, truth - pred)]
        for c, (img, t) in enumerate(zip(imgs, titles)):
            ax[r, c].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(t)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"saved panels -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.path.join(_ROOT, "data", "synthetic", "seq_train"))
    p.add_argument("--ckpt", default=None, help="defaults to weights/<dataset>.pt")
    p.add_argument("--max-windows", type=int, default=300)
    args = p.parse_args()
    ckpt = args.ckpt or os.path.join(WEIGHTS_DIR,
                                     os.path.basename(os.path.normpath(args.data)) + ".pt")
    evaluate(args.data, ckpt, max_windows=args.max_windows)


if __name__ == "__main__":
    main()
