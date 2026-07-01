#!/usr/bin/env bash
# Convenience runner for the SH-WFS prototype.
#   ./run.sh           -> Method 1 demo (writes outputs/reconstruction.png)
#   ./run.sh test      -> fast regression suite (every push)
#   ./run.sh test-all  -> full suite incl. slow Monte-Carlo + golden vectors
#   ./run.sh validate  -> visual validation report (validations/outputs/index.html)
#   ./run.sh dataset     -> generate an i.i.d. ensemble dataset into data/synthetic/
#   ./run.sh seqdataset  -> generate a temporal (N,T,D,D) dataset into data/synthetic/
#   ./run.sh benchmark   -> compare methods on a dataset (pass --data ... [--resunet ckpt])
#   ./run.sh latency     -> end-to-end per-frame latency (real-time budget check)
set -euo pipefail
cd "$(dirname "$0")"

cmd="${1:-demo}"; shift || true
case "$cmd" in
  demo) python3 methods/modal_zernike/demo.py ;;
  test) python3 -m pytest -m fast -q ;;
  test-all) python3 -m pytest -q ;;
  validate) python3 validations/report.py ;;
  dataset) python3 data/make_dataset.py "$@" ;;
  seqdataset) python3 data/make_sequence_dataset.py "$@" ;;
  benchmark) python3 validations/benchmark.py "$@" ;;
  latency) python3 latencycheck.py "$@" ;;
  *) echo "usage: ./run.sh [demo|test|test-all|validate|dataset|seqdataset|benchmark|latency]"; exit 1 ;;
esac
