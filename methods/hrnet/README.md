# Method 4 — HRNet with progressive temporal injection (baseline model)

**Scope:** the architecture only (`model.py`). Data flow, dataset wiring,
training, and inference are owned separately.

## Temporal scheme
Inputs are frames **t, t‑1, …, t‑(n_frames‑1)** (current frame first); for
**`n_frames=5`** that is **t plus the previous 4** (t, t‑1, t‑2, t‑3, t‑4). The
target is the **wavefront at t**.

```
stem(t) → [block1] → +t-1 → [block2] → +t-2 → [block3] → +t-3 → [block4] → +t-4 → [block5] → head
                                          ↑ skip          ↑ skip          ↑ skip
                              (from the PREVIOUS block's start, rolling)
```
- frame **t** seeds the network;
- after each **big HRNet block** (one stage = ResNet `BasicBlock`s per branch +
  cross-scale fusion) the next older frame is injected (added) into the
  highest-resolution branch;
- a **skip connection feeds the previous block's branch‑0 input into the current
  block** — a rolling chain, *not* the global start broadcast to everything.
  Block 2 has no skip (its predecessor's start *is* the global start, excluded);
  blocks 3+ skip from their immediate predecessor;
- a **LayerNorm** (`GroupNorm(1,C)`, size-agnostic) is applied right after each
  temporal connection (the injection + skip merge);
- later-injected frames pass through fewer main blocks, so each frame's **encoder
  depth grows with its lag** (the last frame is the deepest) to compensate —
  tuned by `frame_depth_growth`.

Multi-resolution **parallel branches** (HRNet) are retained: streams grow
`[1]→[1,2]→[1,2,3]…` over the stages and fuse every block. `n_frames` is the only
temporal knob and is decoupled from `n_branches`.

## I/O contract
```python
from methods.hrnet import HRNetWavefront

net = HRNetWavefront(n_frames=5, base_channels=18, n_branches=3,
                     head="map", out_size=48)
y = net(x)        # x: (B, 5, H, W)  ->  y: (B, 1, 48, 48) in [-1,1]
```
- **input** `(B, n_frames, H, W)`: channel 0 = **t**, channel `i` = **t‑i**
  (`n_frames=5` → 5 channels: t, t‑1, t‑2, t‑3, t‑4). Feed from a `WindowView`
  with the current frame first. Size-agnostic.
- **output**: `head="map"` → `(B,1,out_size,out_size)` tanh‑bounded wavefront at t
  (matches the dataset's scaled `wavefronts.npy`; ×amplitude → radians);
  `head="coeffs"` → `(B, n_coeffs)`.

## Knobs
| arg | meaning | default |
|---|---|---|
| `n_frames` | total input frames t … t‑(n_frames‑1); one big block each | 5 |
| `n_branches` | parallel resolution streams (grow over first stages) | 3 |
| `base_channels` | width of the highest-res branch (branches double) | 18 |
| `blocks_per_branch` | ResNet blocks per branch per stage | 2 |
| `frame_depth_growth` | extra encoder blocks per lag step (last frame deepest); 0 = flat | 1 |
| `head` / `out_size` / `n_coeffs` | `"map"`\|`"coeffs"` / map size / coeff count | map / 48 / 20 |

## Verified
- `n_frames` = 2,3,5,7 → output `(B,1,48,48)`; `branches_at` grows `[1,2,3,3,3,…]`;
  ~0.90 M params at `base_channels=16, n_frames=5`.
- Per-frame encoder depths `[2,3,4,5,6]` (last frame deepest) at `frame_depth_growth=1`;
  4 LayerNorms after the temporal connections.
- Both heads run; size-agnostic (64→16); trains (overfits a tiny batch 0.47→0.016).
  `pick_device()` provided.
