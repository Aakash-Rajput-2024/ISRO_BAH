# Method 3 — Deep-learning (ResU-Net) wavefront reconstruction

A PyTorch reimplementation of:

> Z. A. Noel, T. J. Bukowski, S. Gordeyev, R. M. Rennie, *"Shack-Hartmann
> wavefront reconstruction by deep learning neural network for adaptive optics,"*
> Proc. SPIE 12693, Unconventional Imaging, Sensing, and Adaptive Optics 2023,
> 126930G. doi:10.1117/12.2677670

A Res-UNet CNN maps a **raw SH-WFS frame directly to a wavefront**, bypassing
both centroiding and least-squares matrix inversion — the two steps the paper
identifies as the speed bottleneck for closed-loop AO.

## Faithful to the paper
| Paper element | Here |
|---|---|
| Multi-kernel residual blocks (3×3/5×5/7×7/9×9 → concat → 1×1 fuse → BN → residual add) | `model.py: MultiKernelResBlock` |
| Stride-4 conv downsampling (not max-pool); features double each level (24→48→96) | `model.py: ResUNet` |
| Transposed-conv upsampling + U-Net skip concatenation | `model.py: ResUNet` |
| FC head with **tanh** (wavefront can be negative) + resize to target grid | `model.py: ResUNet.head` |
| Zonal output (a wavefront map, not Zernike modes) | coarse `target×target` map |
| "Average-added" preprocessing (Fig. 4/5): time-averaged spot added per frame | `preprocess.py: average_added` |
| ADAM, lr 1e-4, MSE loss on the scaled wavefront (eq. 1) | `train.py` |
| Training/validation loss curve (Fig. 6) | `train.py: _plot_loss` |

## Two deliberate adaptations
1. **PyTorch, not TensorFlow** — torch 2.x with Apple-MPS / CUDA acceleration is
   what's available here; the architecture is unchanged.
2. **Trained on the synthetic atmospheric dataset**, not the paper's wind-tunnel
   data (which we don't have, and which is aero-optical). The ISRO problem is
   *atmospheric*, and `data/make_dataset.py` gives labelled (frame → wavefront)
   pairs with exact ground truth — the right training signal for this task.

## Files
| file | role |
|---|---|
| `model.py` | `ResUNet` architecture + device picker |
| `preprocess.py` | intensity normalisation + average-added transform |
| `dataset.py` | load `data/synthetic/<name>`, build coarse [-1,1] wavefront targets |
| `train.py` | training loop → checkpoint (`weights/`) + loss curve (`outputs/`) |
| `reconstruct.py` | `ResUNetReconstructor.process(frame)` → `.wavefront`, `.coeffs` |

## Usage
```bash
# 1. generate a labelled dataset (ground-truth wavefronts)
python data/make_dataset.py --n 2000 --name train_dr0_6

# 2. train (MPS/CUDA auto-detected)
python methods/deep_resunet/train.py --data data/synthetic/train_dr0_6 --epochs 25

# 3. reconstruct (same interface as Method 1)
python - <<'PY'
import numpy as np
from methods.deep_resunet import ResUNetReconstructor
rec = ResUNetReconstructor("methods/deep_resunet/weights/train_dr0_6.pt")
res = rec.process(np.load("data/synthetic/train_dr0_6/frames.npy")[0])
print(res.wavefront.shape, res.coeffs.shape)
PY
```

`process()` returns a pupil wavefront [rad] (the coarse zonal map upsampled,
piston-removed) and Zernike coefficients (by projecting the predicted wavefront
onto the Noll basis), so Method 3 plugs into the same validation plots and any
method-comparison benchmark as Method 1.

## Status & expectations
This is a faithful reimplementation, not a tuned production model. As the paper
itself reports, an under-trained ResU-Net produces large errors (their absolute
error was ~the same magnitude as the signal). Accuracy needs a large dataset and
real training; the smoke tests (`pytest -m ml`) only verify the architecture is
correct and *can* learn (forward shape + overfits a tiny batch). For the ISRO
evaluation, Method 1 (modal) remains the primary deliverable — Method 3 is the
ML comparison point.
