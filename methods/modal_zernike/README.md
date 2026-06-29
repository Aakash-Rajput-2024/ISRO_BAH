# Method 1 — Modal (Zernike) wavefront reconstruction

The baseline reconstruction method. The wavefront is expanded in Noll-ordered
Zernike polynomials, `W = Σ_k a_k Z_k`, so the measured sub-aperture slopes are
linear in the coefficients:

```
s = D · a            D = modal interaction matrix  (2·n_valid × n_modes)
a = pinv(D) · s      least-squares reconstruction (R = pinv(D) precomputed once)
```

Per frame the reconstruction is a single matrix-vector product `R @ s` — the
step targeted for the eventual C port (validated by the golden vectors in
`validations/`).

## Files
| file | role |
|---|---|
| [`reconstruct.py`](reconstruct.py) | builds `D`, precomputes `R = pinv(D)`, maps slopes→coeffs→wavefront |
| [`pipeline.py`](pipeline.py) | `WFSPipeline`: calibrate (reference spots) + per-frame `process()` |
| [`demo.py`](demo.py) | end-to-end demo → `outputs/reconstruction.png` |

Reuses the shared core in [`methods/common/`](../common/) for config, pupil
geometry, the Zernike basis, the phase-screen simulator, centroiding, and the
`r0`/`tau0` estimators.

## Usage
```python
from methods.modal_zernike import Config, WFSPipeline
from methods.common.simulate import flat_frame

pipe = WFSPipeline(Config())
pipe.calibrate(flat_frame(pipe.cfg, pipe.geom))   # set reference spot positions
result = pipe.process(frame)                       # .slopes .coeffs .wavefront
```

## Scope / known limits
- Modal Zernike suits **atmospheric** (isotropic, on-axis) turbulence — the ISRO
  problem. It is a poor fit for aero-optical, laterally-translating aberrations
  (Noel et al. 2023), where a zonal/Southwell or learned method would be added
  as a sibling under `methods/`.
- The FFT phase screen under-represents low spatial frequencies, biasing tip/tilt
  and absolute `r0` (documented and measured in `validations/`).

## Not yet implemented (ISRO deliverable gap)
The **deformable-mirror actuator map** in actuator-stroke units, with
inter-actuator coupling (`c = pinv(H) · (−W/2)`), is still outstanding.
