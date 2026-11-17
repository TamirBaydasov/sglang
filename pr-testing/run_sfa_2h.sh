#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
sha256sum --check CHECKSUMS.sha256
exec python3 -u "$SCRIPT_DIR/sfa2h.py" "$@"
