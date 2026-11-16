#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 -u "$SCRIPT_DIR/session10.py" \
  --hours 10 --round all --case-minutes 110 --accuracy-minutes 90 --speed-minutes 20 \
  --moe-mode tp --accuracy-server-requests 1 --decode-graph off --memfrac 0.80 \
  --baseline-ref aa32b34a4aec8b6bea740d20d520e3f6a4ff03d6 --no-fetch \
  --gsm8k-samples 64 --gsm8k-parallel 16 --gsm8k-max-tokens 1024 \
  --gsm8k-timeout 1800 --gsm8k-min-accuracy 0.90 \
  --save-dir /home/swx1199799 "$@"
