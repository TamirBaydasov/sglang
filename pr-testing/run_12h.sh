#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 -u "$SCRIPT_DIR/run_pr_tests.py" \
  --suite overnight12 --round all --memfrac 0.76 \
  --save-dir /home/swx1199799 "$@"
