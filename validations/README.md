# validations/

Proves the methods in `methods/` are correct — not just self-consistent — and
keeps them correct. Generates a visual report and a pass/fail metrics table.

## Run
```bash
./run.sh test       # fast regression gate          (pytest -m fast)
./run.sh test-all   # full suite                     (pytest)
./run.sh validate   # visual report -> validations/outputs/index.html

pytest -m slow      # Monte-Carlo characterisation
pytest -m crossval  # vs aotools/hcipy (skips if absent)
pytest -m golden    # C-port oracle conformance
pytest -m hardware  # real-data checks (skips until fixtures present)
```

## Layout
| file | role |
|---|---|
| `metrics.py` | plot-free numeric kernels (structure fn, PSD, cross-talk, residual RMS, Noll/Kolmogorov theory) — shared by tests **and** report |
| `plots.py` | matplotlib figure builders |
| `thresholds.py` | canonical acceptance thresholds (`TOL`) — single source of truth |
| `oracle.py` | golden-vector dump/load + tolerance comparator (C-port equivalence) |
| `references.py` | lazy adapters to independent libraries (aotools / hcipy) |
| `realdata.py` | real lab-frame ingest + sub-aperture grid auto-detection |
| `report.py` | runs the characterisations → `outputs/index.html` (~14 figures + table) |
| `gen_golden_vectors.py` | regenerate the Tier-4 fixtures (then commit) |
| `tests/` | the pytest suite + `fixtures/golden/` |

## Four tiers
1. **Synthetic rigor** — fidelity, estimator bias/variance, phase-screen statistics vs theory.
2. **Cross-validation** — parity with an *independent* implementation (breaks self-consistency).
3. **Real lab-data path** — grid auto-detection (tested on synthetic now), known-aberration recovery (on real frames).
4. **C-port equivalence** — golden I/O + stored `D`/`R`, compared with `oracle.compare`.

Methodology, thresholds, and the data/C-port contracts are in
[`docs/VALIDATION.md`](../docs/VALIDATION.md).
