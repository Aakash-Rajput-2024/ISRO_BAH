# SH-WFS wavefront reconstruction & turbulence characterisation

Prototype for the ISRO problem statement: reconstruct wavefronts from
Shack-Hartmann wavefront sensor (SH-WFS) time-series frames, characterise
turbulence (Fried parameter `r0`, coherence time `tau0`), and derive deformable
mirror actuator maps.

This Python package is the **reference implementation / test oracle**. The
performance-critical inner loop (centroiding + matrix-vector reconstruction) is
intended to be ported to C later; this code validates that port and lets us
iterate on the algorithms fast. It runs **fully synthetic** — no lab data needed
yet — by generating Kolmogorov phase screens with a known `r0`, so every stage
can be checked against ground truth.

## Quick start

```bash
pip install -r requirements.txt
./run.sh          # end-to-end demo -> outputs/reconstruction.png
./run.sh test     # fast regression suite
./run.sh test-all # full suite (slow Monte-Carlo + golden + cross-validation)
./run.sh validate # visual validation report -> outputs/validation/index.html
./run.sh latency  # end-to-end per-frame latency, all methods -> outputs/latency.md
```

Validation methodology, acceptance thresholds, and the real-data / C-port
contracts are documented in [docs/VALIDATION.md](docs/VALIDATION.md).

## Repository layout

```
data/                datasets + contract (synthetic generated, real .bmp)   -> data/README.md
methods/
  common/            method-agnostic core (config, geometry, zernike,
                     phasescreen, simulate, centroid, turbulence)
  modal_zernike/     Method 1: modal Zernike reconstruction               -> methods/modal_zernike/README.md
  deep_resunet/      Method 3: deep-learning ResU-Net (Noel et al. 2023)  -> methods/deep_resunet/README.md
validations/         metrics, plots, oracle, report + the pytest suite     -> validations/README.md
docs/VALIDATION.md   validation methodology & thresholds
```

Add future reconstruction methods (zonal/Southwell, Karhunen-Loève, learned) as
sibling sub-packages under `methods/`, reusing `methods/common/`.

## Pipeline

```
WFS frame  ->  centroiding  ->  slopes  ->  modal reconstruction  ->  W(x,y), Zernike aₖ
                                                                          |
                                          +-------------------------------+
                                          |                               |
                                   turbulence (r0, tau0)          actuator map (DM)   [TODO]
```

| Module | Role |
|---|---|
| `methods/common/config.py` | System parameters (MLA, detector, pupil). Replace defaults with lab numbers. |
| `methods/common/geometry.py` | Pupil grid, valid sub-apertures, per-sub-aperture averaging operator. |
| `methods/common/zernike.py` | Noll-ordered Zernike polynomials (unit-variance normalisation). |
| `methods/common/phasescreen.py` | Kolmogorov phase screens (FFT) + frozen-flow time series. |
| `methods/common/simulate.py` | Synthetic SH-WFS frame generator (the test oracle). |
| `methods/common/centroid.py` | Thresholded centre-of-gravity spot centroiding. |
| `methods/common/turbulence.py` | `r0` from Zernike-variance (Noll), `tau0` from temporal autocorrelation. |
| `methods/modal_zernike/reconstruct.py` | Modal interaction matrix `D`, reconstructor `pinv(D)`. |
| `methods/modal_zernike/pipeline.py` | Glue: calibrate + per-frame processing. |

## Key physics

- The SH-WFS measures the **local wavefront slope** per sub-aperture, not the
  phase directly: `spot_shift = f * slope` (f = lenslet focal length).
- Reconstruction is **modal** (Zernike): `slopes = D · a`, solved as
  `a = pinv(D) · slopes`. `D` and `pinv(D)` are precomputed once; per frame it is
  a single matrix-vector product — the step to optimise in C.
- `r0` from the variance of mid-order Zernike coefficients vs the Kolmogorov
  model (Noll 1976). `tau0` from the 1/e decay of the wavefront's temporal
  autocorrelation across the frame series.

## Status / next steps

Done: synthetic generator, centroiding, slopes, modal reconstruction, `r0`,
`tau0`, ground-truth tests, **C++ centroiding inner loop** (`methods/common/cpp/`,
golden-tested vs numpy) and an **end-to-end latency benchmark** (`latencycheck.py`).

Planned:
- Zonal (Fried-geometry) reconstruction as an alternative to modal.
- Actuator map: build the influence-function matrix `H` (with inter-actuator
  coupling), solve `c = pinv(H) · (-W/2)` in actuator-stroke units.
- Subharmonic-augmented phase screens (removes the FFT low-frequency `r0` bias).
- Real `.bmp` ingest + sub-aperture grid auto-detection from a flat frame.
- Port the modal matrix-vector reconstruction to C++ too (centroiding is done).

## Latency (real-time budget)

The control loop must produce a wavefront within ~10 ms of a frame arriving.
`latencycheck.py` times the **full** per-frame chain (not just the matrix
product) for every method on identical, realistically-rendered frames, and
reports mean/median/p95/p99 + frame rate against the budget:

| method (per frame) | mean | p99 | vs 10 ms |
|---|---|---|---|
| modal_zernike (numpy) | 2.1 ms | 2.3 ms | **PASS** |
| modal_zernike_cpp | 0.5 ms | 0.6 ms | **PASS** |
| deep_resunet (Method 3) | 31 ms | 34 ms | FAIL |
| hrnet_full (18ch, blk2, grow1; 1.14 M) | 13.7 ms | 14.4 ms | FAIL |
| hrnet_small (18ch, blk1, grow0; 648 k) | 8.7 ms | 9.4 ms | **PASS** |

Numbers are from an Apple-MPS dev box (idle); run `./run.sh latency` for your
hardware. Two changes bring a method under budget:

- **C++ centroiding inner loop** (`methods/common/cpp/`): the per-sub-aperture
  pixel loop drops from ~1.8 ms to ~0.25 ms (~7×), so the whole modal path is
  ~0.5 ms — and it stays the most *accurate* method too (held-out R² ≈ 0.94).
- **Depth-trimmed HRNet** (`hrnet_small`): 13.7 → 8.7 ms (1.6×), 648 k vs
  1.14 M params. It was shrunk by cutting **sequential depth** (blocks/branch,
  per-frame encoder growth), *not* channels — on this GPU the net is
  launch/depth-bound, so channels barely move latency (18→32 ch ≈ same ms).
  Cost: held-out wavefront R² 0.836 → 0.772 (−7.6%); channel width was kept
  full precisely to limit that loss. Retrain with
  `methods/hrnet/train.py --channels 18 --blocks-per-branch 1 --frame-depth-growth 0`.

## Caveats

- Defaults in `config.py` are placeholders; swap in the real MLA pitch/focal
  length, pixel size, pupil diameter, and DM coupling when available.
- The plain FFT phase screen under-represents the lowest spatial frequencies, so
  absolute `r0` carries a known bias (~10–20% here); the estimator uses mid-order
  modes to limit it.
