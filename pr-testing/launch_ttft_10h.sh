#!/usr/bin/env bash
set -euo pipefail
TTFT10_STARTED_MONOTONIC=$(python3 -c 'import time; print(time.monotonic())')
export TTFT10_STARTED_MONOTONIC
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
sha256sum --check CHECKSUMS.sha256
# Validate the full experiment before touching the container or opening screen.
bash ./run_ttft_10h.sh "$@" --dry-run > /dev/null
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then
    exec bash ./run_ttft_10h.sh "$@"
  fi
done
command -v screen >/dev/null
python3 ./prepare_ttft10_sources.py "$@"
SESSION=glm-ttft10-$(date -u +%Y%m%d-%H%M%S)-$$
CONSOLE="$SCRIPT_DIR/$SESSION.log"
screen -L -Logfile "$CONSOLE" -dmS "$SESSION" bash ./run_ttft_10h.sh "$@"
printf 'Attach: screen -r %s\nConsole: %s\nResults default: /home/swx1199799/glm-ttft10-results/\n' "$SESSION" "$CONSOLE"
