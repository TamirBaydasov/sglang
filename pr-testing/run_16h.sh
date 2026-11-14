#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CASES=00-baseline,14-baseline-ep,09-pr1-megamoe,01-pr-e-chat-encode,02-pr-b-indexer-int8
CASES+=,04-pr-f-sfa-kv-8bit,18-baseline-prefill1,03-pr-prefetch-packed,07-combined,19-baseline-prefill1-repeat
CASES+=,05-pr-ts-token-sharded,06-pr-moe-int8
CASES+=,20-moe-ts-off-eager,21-moe-ts-on-eager,22-moe-ts-off-graph,23-moe-ts-on-graph,10-ep-p4-eager
CASES+=,15-baseline-8k,16-megamoe-8k,17-baseline-8k-repeat,08-baseline-repeat
exec python3 -u "$SCRIPT_DIR/run_pr_tests.py" \
  --round all --cases "$CASES" --hours 16 --case-minutes 45 \
  --decode-graph off --memfrac 0.70 \
  --baseline-ref npu-w4a8-tp-bias-fix \
  --gsm8k-samples 64 --gsm8k-parallel 16 --gsm8k-max-tokens 1024 \
  --gsm8k-min-accuracy 0.90 --save-dir /home/swx1199799 "$@"
