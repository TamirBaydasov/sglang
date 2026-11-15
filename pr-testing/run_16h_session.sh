#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 -u "$SCRIPT_DIR/session16.py" \
  --hours 16 --round all --case-minutes 80 --accuracy-minutes 60 --speed-minutes 20 \
  --decode-graph off --memfrac 0.70 --baseline-ref npu-w4a8-tp-bias-fix \
  --gsm8k-samples 64 --gsm8k-parallel 16 --gsm8k-max-tokens 1024 \
  --gsm8k-timeout 1800 --gsm8k-min-accuracy 0.90 \
  --save-dir /home/swx1199799 "$@"
