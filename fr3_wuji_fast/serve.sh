#!/usr/bin/env bash
set -euo pipefail
PI05_FAST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_FAST_DIR/../deploy.sh" serve normal "$@"
