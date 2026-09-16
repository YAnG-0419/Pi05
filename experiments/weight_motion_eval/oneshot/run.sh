#!/usr/bin/env bash
set -euo pipefail
PI05_ONESHOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_ONESHOT_DIR/../../../deploy.sh" slow54 "$@"
