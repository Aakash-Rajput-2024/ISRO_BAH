#!/usr/bin/env bash
# Convenience runner for the SH-WFS prototype.
#   ./run.sh         -> run the end-to-end demo (writes outputs/reconstruction.png)
#   ./run.sh test    -> run the ground-truth test suite
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-demo}" in
  demo) python3 scripts/demo.py ;;
  test) python3 -m pytest tests/ -q ;;
  *) echo "usage: ./run.sh [demo|test]"; exit 1 ;;
esac
