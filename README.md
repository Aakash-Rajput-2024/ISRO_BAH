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
`tau0`, ground-truth tests.

Planned:
- Zonal (Fried-geometry) reconstruction as an alternative to modal.
- Actuator map: build the influence-function matrix `H` (with inter-actuator
  coupling), solve `c = pinv(H) · (-W/2)` in actuator-stroke units.
- Subharmonic-augmented phase screens (removes the FFT low-frequency `r0` bias).
- Real `.bmp` ingest + sub-aperture grid auto-detection from a flat frame.
- C port of the centroiding + reconstruction inner loop; benchmark < ~10 ms/frame.

## Caveats

- Defaults in `config.py` are placeholders; swap in the real MLA pitch/focal
  length, pixel size, pupil diameter, and DM coupling when available.
- The plain FFT phase screen under-represents the lowest spatial frequencies, so
  absolute `r0` carries a known bias (~10–20% here); the estimator uses mid-order
  modes to limit it.
