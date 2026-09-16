#!/usr/bin/env bash
set -euo pipefail
PI05_ADAPTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_ADAPTER_DIR/../fr3_wuji/run.sh" "$PI05_ADAPTER_DIR/serve.py" "$@"
