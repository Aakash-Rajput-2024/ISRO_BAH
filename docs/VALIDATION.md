# Validation methodology

How we prove the SH-WFS models are correct — not just self-consistent — and
keep them correct as the code evolves toward a C port and real lab frames.

The pipeline's forward model (`simulate.py`) and inverse model (`reconstruct.py`)
deliberately share one averaging operator, so a noiseless round-trip is exact by
construction. That makes round-trip tests necessary but **not sufficient**: a bug
present in both directions would pass silently. Validation is therefore layered
in four tiers, the first two of which break that circularity.

## Tiers

| Tier | What it proves | Where |
|---|---|---|
| **1 Synthetic rigor** | Models behave correctly & predictably across SNR, turbulence strength, mode count | `test_reconstruction_fidelity.py`, `test_estimators.py`, `test_phasescreen_stats.py` |
| **2 Cross-validation** | Our math matches an *independent* implementation (not just itself) | `test_crossvalidation.py` (aotools / hcipy) |
| **3 Real lab-data path** | Grid detection + recovery of physically-injected aberrations | `test_realdata.py`, `validations/realdata.py` |
| **4 C-port equivalence** | A future C inner loop reproduces the Python oracle bit-for-bit (within tol) | `test_golden_vectors.py`, `validations/oracle.py` |

## Running

```bash
./run.sh test           # fast suite (every push, < ~10 s): pytest -m fast
./run.sh test-all       # everything incl. slow Monte-Carlo + golden + crossval
./run.sh validate       # visual report -> validations/outputs/index.html

pytest -m slow          # Monte-Carlo characterisation
pytest -m crossval      # vs aotools/hcipy (skips if not installed)
pytest -m golden        # C-port oracle conformance
pytest -m hardware      # real-data checks (skips until fixtures present)

pip install -r requirements-dev.txt   # to enable crossval (aotools/hcipy)
python validations/gen_golden_vectors.py  # regenerate Tier-4 fixtures (then commit)
```

## Markers

`fast` (push-gate), `slow` (nightly Monte-Carlo), `crossval`, `hardware`, `golden`.

## Acceptance thresholds

All thresholds live in one place — `validations/thresholds.py` (`TOL`) —
shared by the test suite and the report.

| Check | Threshold | Rationale |
|---|---|---|
| Reconstructor left-inverse `R@D` | `|R@D − I|` < 1e-9 | D has full column rank; `pinv(D)@D = I` |
| Empirical mode cross-talk | off-diag < 0.05, gain ∈ [0.95, 1.05] | near-orthogonal reconstruction incl. centroiding |
| Single-mode recovery (noiseless) | < 0.03 rad | matches the original `test_zernike_recovery` |
| r0 relative error | < 40 % | the FFT screen's low-frequency deficit is real (below) |
| Structure-fn slope (mid-range) | within 0.25 of 5/3 | validate the **shape**, not the deficit-biased magnitude |
| PSD slope (mid-range) | within 0.30 of −11/3 | as above |
| Tip/tilt variance vs Noll | < 0.20 | quantifies why `estimate_r0` skips j=2,3 |
| Mid-order captured fraction | ∈ [0.50, 1.10] | j≥4 recovers most of the Noll-predicted variance |
| tau0·wind invariance | < 5 % spread | frozen-flow: tau0 ∝ 1/wind (robust check) |
| Golden centroids / coeffs | < 1e-6 px / rel < 1e-9 | numpy determinism using the **stored** R |

### Known bias — document, don't hide

The plain FFT phase screen (`phasescreen.py`) under-represents the lowest spatial
frequencies because the screen period equals the pupil size (Lane, Glindemann &
Dainty 1992). Consequences, all visible in the report:

- the structure function falls **below** `6.88(r/r0)^(5/3)` at large separations;
- tip/tilt (Noll j=2,3) variance is suppressed to a few % of theory;
- absolute r0 carries a bias, so `estimate_r0` fits **mid-order** modes (j≥4).

We therefore validate power-law **slopes** and the mid-order band, not absolute
low-frequency magnitudes. Subharmonic augmentation (a planned `phasescreen.py`
refinement) would tighten these and let the thresholds shrink.

## Real lab-data contract (Tier 3)

Drop fixtures into `validations/tests/fixtures/realdata/` to activate `-m hardware`:

- `flat.{npy,bmp,png,fits}` — flat-wavefront calibration frame.
- `defocus.npy` + `defocus_truth.npy` — a frame with a physically-injected,
  known aberration and its commanded Zernike-coefficient vector.
- Grayscale intensity, background **not** subtracted, square detector, one MLA
  spanning the pupil. Update `config.py` with the real MLA pitch / focal length /
  pixel size / pupil diameter before running.

`validations/realdata.py` auto-detects the sub-aperture grid (pitch via
projection autocorrelation, lenslet count via peak finding); this is validated
today against the synthetic flat frame and runs unchanged on real frames.

## C-port equivalence (Tier 4)

`validations/gen_golden_vectors.py` writes canonical per-stage I/O (frame → centroids
→ slopes → coeffs) plus the operators **D and R** and a config-hash manifest to
`validations/tests/fixtures/golden/`. The C port must consume the stored **R** rather than
recompute `pinv` (SVD is not bit-stable across BLAS builds), and is diffed
against these fixtures via `oracle.compare(...)`.
