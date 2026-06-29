"""Train the ResU-Net (Method 3) on a synthetic dataset.

    # first generate data (has ground-truth wavefronts):
    python data/make_dataset.py --n 400 --name train_dr0_6
    # then train:
    python methods/deep_resunet/train.py --data data/synthetic/train_dr0_6 --epochs 25

Saves a checkpoint (weights + the metadata inference needs) and a training/
validation loss curve (cf. Noel et al. Fig. 6). ADAM, lr 1e-4, MSE loss on the
[-1,1]-scaled wavefront -- as in the paper.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from methods.deep_resunet.dataset import load_synthetic, split
from methods.deep_resunet.model import ResUNet, pick_device

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def train(data_path, epochs=25, batch=8, lr=1e-4, target=48, base=24,
          avg_added=True, device=None, seed=0):
    torch.manual_seed(seed)
    device = device or pick_device()
    X, Y, meta = load_synthetic(data_path, target=target, avg_added=avg_added)
    (Xt, Yt), (Xv, Yv) = split(X, Y, seed=seed)
    tl = DataLoader(TensorDataset(Xt, Yt), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(Xv, Yv), batch_size=batch)

    model = ResUNet(target=target, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train(); tr = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tr += loss.item() * xb.size(0)
        model.eval(); va = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                va += loss_fn(model(xb), yb).item() * xb.size(0)
        tr /= len(tl.dataset); va /= len(vl.dataset)
        hist["train"].append(tr); hist["val"].append(va)
        print(f"epoch {ep+1:3d}/{epochs}  train {tr:.4e}  val {va:.4e}")

    ckpt = {
        "state_dict": model.state_dict(),
        "target": target, "base": base,
        "wf_scale": meta["wf_scale"], "npix": meta["npix"],
        "avg_added": avg_added,
        "mean_frame": meta["mean_frame"],
        "history": hist,
    }
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    name = os.path.basename(os.path.normpath(data_path))
    path = os.path.join(WEIGHTS_DIR, f"{name}.pt")
    torch.save(ckpt, path)
    print(f"saved checkpoint -> {path}")
    _plot_loss(hist, name)
    return path, hist


def _plot_loss(hist, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(hist["train"], "--", label="training loss")
    ax.plot(hist["val"], "-", label="validation loss")
    ax.set_xlabel("epoch"); ax.set_ylabel(r"MSE loss (scaled wavefront)")
    ax.set_title("ResU-Net training (Method 3)"); ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"loss_{name}.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"saved loss curve -> {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="data/synthetic/<name> folder")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target", type=int, default=48)
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--no-avg-added", action="store_true")
    args = p.parse_args()
    train(args.data, epochs=args.epochs, batch=args.batch, lr=args.lr,
          target=args.target, base=args.base, avg_added=not args.no_avg_added)


if __name__ == "__main__":
    main()
