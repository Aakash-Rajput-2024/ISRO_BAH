"""Ablation study -- validate that the architecture's choices earn their keep.

The headline check: does temporal history actually help? Trains the model at
several `n_frames` (1 = current frame only, vs 5 = t + previous 4) on the SAME
dataset and split, then reports held-out wavefront R^2 for each. If R^2 does not
improve as frames are added, the temporal design is not justified -- this is the
scientific validation of the model, not just a performance number.

    python methods/hrnet/ablate.py --epochs 20
    python methods/hrnet/ablate.py --frames 1 3 5 --epochs 20

(Each setting is a separate training run; use a real epoch count for conclusions.)
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from methods.hrnet.train import train, resolve_data
from methods.hrnet.eval import evaluate


def run(data_path, frames, epochs, channels, batch, max_windows):
    results = []
    for nf in frames:
        print(f"\n=== n_frames={nf} ===")
        ckpt = train(data_path, n_frames=nf, channels=channels, epochs=epochs,
                     batch=batch, tag=f"_n{nf}")
        rep = evaluate(data_path, ckpt, max_windows=max_windows, n_panels=0)
        r2, rms, strehl = rep["hrnet"]
        results.append((nf, r2, rms, strehl))

    base = results[0][1]                       # R^2 at the fewest frames
    print("\n" + "=" * 60)
    print(f"{'n_frames':>9} | {'wavefront R²':>13} | {'resid RMS':>10} | {'ΔR² vs n=' + str(frames[0]):>14}")
    print("-" * 60)
    for nf, r2, rms, _ in results:
        print(f"{nf:>9} | {r2:>+13.3f} | {rms:>10.3f} | {r2 - base:>+14.3f}")
    best = max(results, key=lambda r: r[1])
    print(f"\nbest: n_frames={best[0]}  (R²={best[1]:+.3f})")
    print("temporal history helps" if best[0] != frames[0]
          else "temporal history did NOT help at this training budget")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=None)
    p.add_argument("--frames", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--channels", type=int, default=18)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=300)
    args = p.parse_args()
    data_path = resolve_data(args.data)
    run(data_path, args.frames, args.epochs, args.channels, args.batch, args.max_windows)


if __name__ == "__main__":
    main()
