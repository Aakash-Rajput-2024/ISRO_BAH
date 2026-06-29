#!/usr/bin/env bash
# Convenience runner for the SH-WFS prototype.
#   ./run.sh           -> Method 1 demo (writes outputs/reconstruction.png)
#   ./run.sh test      -> fast regression suite (every push)
#   ./run.sh test-all  -> full suite incl. slow Monte-Carlo + golden vectors
#   ./run.sh validate  -> visual validation report (validations/outputs/index.html)
#   ./run.sh dataset   -> generate a synthetic dataset into data/synthetic/
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-demo}" in
  demo) python3 methods/modal_zernike/demo.py ;;
  test) python3 -m pytest -m fast -q ;;
  test-all) python3 -m pytest -q ;;
  validate) python3 validations/report.py ;;
  dataset) python3 data/make_dataset.py ;;
  *) echo "usage: ./run.sh [demo|test|test-all|validate|dataset]"; exit 1 ;;
esac
