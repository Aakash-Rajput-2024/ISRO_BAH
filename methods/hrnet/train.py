"""Train Method 4 (HRNet temporal) on a (N, T, D, D) sequence dataset.

    python data/make_sequence_dataset.py --n 120 --t 24 --name seq_train
    python methods/hrnet/train.py --data data/synthetic/seq_train --epochs 20

Feeds each model input as frames [t, t-1, ..., t-(n_frames-1)] (current first)
and the target as the wavefront at t (downsampled, scaled to [-1,1] to match the
tanh map head). Splitting is by instance (from the dataset manifest), so windows
never leak across train/val. Saves a checkpoint + a train/val loss curve.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:                       # graceful fallback if tqdm is absent
    def tqdm(it, **kw):
        return it

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from methods.common.sequence_dataset import load_sequences, split_from_manifest
from methods.hrnet import HRNetWavefront, pick_device

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DEFAULT_DATA = os.path.join(_ROOT, "data", "synthetic", "seq_train")


def resolve_data(path, auto_n=120, auto_t=24, seed=1):
    """Use the given sequence dataset, else reuse/auto-generate the default one."""
    import json
    if path is None:
        path = DEFAULT_DATA
    manifest = os.path.join(path, "manifest.json")
    if os.path.exists(manifest):
        if json.load(open(manifest)).get("kind") != "sequence":
            raise SystemExit(f"{path} is not a temporal dataset (kind != 'sequence'); "
                             f"generate one with data/make_sequence_dataset.py")
        print(f"using dataset: {path}")
        return path
    # Nothing there -> generate a default temporal dataset.
    from methods.modal_zernike import Config
    from data.make_sequence_dataset import generate
    print(f"no dataset at {path}; generating a default one (N={auto_n}, T={auto_t})...")
    generate(Config(), path, auto_n, auto_t, seed=seed)
    return path


class WindowDataset(Dataset):
    """Wraps a torch-free WindowView -> (frames current-first, scaled wavefront)."""

    def __init__(self, view, wf_scale):
        self.view = view
        self.wf_scale = wf_scale

    def __len__(self):
        return len(self.view)

    def __getitem__(self, i):
        x, y = self.view[i]                      # x:(K,D,D) in [0,1]; y:(w,w) [rad]
        x = np.array(x[::-1])                     # reverse -> [t, t-1, ...]; copy (pos strides)
        y = np.clip(y / self.wf_scale, -1.0, 1.0)[None]   # (1,w,w) scaled
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def train(data_path, n_frames=5, channels=18, epochs=20, batch=8, lr=1e-3,
          device=None, seed=0, tag="", blocks_per_branch=2, frame_depth_growth=1):
    torch.manual_seed(seed)
    device = device or pick_device()
    data = load_sequences(data_path)
    split = split_from_manifest(data)
    w = int(data.wavefronts.shape[-1])
    if n_frames > data.T:
        raise ValueError(f"n_frames={n_frames} > T={data.T}")

    # Wavefront amplitude scale from the TRAIN split (targets -> [-1,1]).
    tr_wf = np.asarray(data.wavefronts[split["train"]], dtype=np.float32)
    wf_scale = float(np.percentile(np.abs(tr_wf), 99.5)) or 1.0

    tv = data.windows(n_frames, horizon=0, label="wavefronts", instances=split["train"])
    vv = data.windows(n_frames, horizon=0, label="wavefronts", instances=split["val"])
    tl = DataLoader(WindowDataset(tv, wf_scale), batch_size=batch, shuffle=True)
    vl = DataLoader(WindowDataset(vv, wf_scale), batch_size=batch)
    print(f"device={device}  train windows={len(tv)}  val windows={len(vv)}  "
          f"n_frames={n_frames}  channels={channels}  blocks/branch={blocks_per_branch}  "
          f"depth_growth={frame_depth_growth}  out_size={w}  wf_scale={wf_scale:.3f}")

    model = HRNetWavefront(n_frames=n_frames, base_channels=channels,
                           blocks_per_branch=blocks_per_branch,
                           frame_depth_growth=frame_depth_growth,
                           head="map", out_size=w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e3:.1f}k")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    name = os.path.basename(os.path.normpath(data_path)) + tag
    ckpt_path = os.path.join(WEIGHTS_DIR, f"{name}.pt")

    def save(epoch, best_val):
        # blocks_per_branch/frame_depth_growth are persisted so eval/inference can
        # rebuild the exact architecture (old checkpoints default to 2/1).
        torch.save({"state_dict": model.state_dict(), "n_frames": n_frames,
                    "channels": channels, "blocks_per_branch": blocks_per_branch,
                    "frame_depth_growth": frame_depth_growth,
                    "out_size": w, "wf_scale": wf_scale, "n_params": n_params,
                    "history": hist, "epoch": epoch, "best_val": best_val}, ckpt_path)

    hist = {"train": [], "val": []}
    best_val = float("inf")
    for ep in range(epochs):
        model.train(); tr = 0.0
        pbar = tqdm(tl, desc=f"epoch {ep+1:3d}/{epochs}", leave=False, unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tr += loss.item() * xb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3e}")
        model.eval(); va = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                va += loss_fn(model(xb), yb).item() * xb.size(0)
        tr /= len(tl.dataset); va /= len(vl.dataset)
        hist["train"].append(tr); hist["val"].append(va)
        # Keep the BEST checkpoint (lowest val), not the last -- the model can
        # mildly overfit past its val minimum, and we want to ship its best.
        improved = va < best_val
        if improved:
            best_val = va
            save(ep + 1, best_val)
        _plot(hist, name)
        print(f"epoch {ep+1:3d}/{epochs}  train {tr:.4e}  val {va:.4e}"
              f"{'  [saved *best]' if improved else ''}")

    print(f"best val {best_val:.4e}  ->  {ckpt_path}")
    return ckpt_path


def _plot(hist, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(hist["train"], "--", label="train")
    ax.plot(hist["val"], "-", label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (scaled wavefront)")
    ax.set_title("HRNet temporal training (Method 4)"); ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"loss_{name}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"saved loss curve -> {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None,
                   help="sequence dataset dir; if omitted, reuse/auto-generate "
                        f"{os.path.relpath(DEFAULT_DATA, _ROOT)}")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--channels", type=int, default=18)
    p.add_argument("--blocks-per-branch", type=int, default=2,
                   help="ResNet blocks per branch per stage (depth knob)")
    p.add_argument("--frame-depth-growth", type=int, default=1,
                   help="extra encoder blocks per frame lag (0 = flat depth)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", default="", help="checkpoint name suffix, e.g. _small")
    p.add_argument("--auto-n", type=int, default=120, help="instances if auto-generating")
    p.add_argument("--auto-t", type=int, default=24, help="frames/instance if auto-generating")
    args = p.parse_args()
    data_path = resolve_data(args.data, auto_n=args.auto_n, auto_t=args.auto_t)
    train(data_path, n_frames=args.n_frames, channels=args.channels,
          blocks_per_branch=args.blocks_per_branch,
          frame_depth_growth=args.frame_depth_growth,
          epochs=args.epochs, batch=args.batch, lr=args.lr, tag=args.tag)


if __name__ == "__main__":
    main()
