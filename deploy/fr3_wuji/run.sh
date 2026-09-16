#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -x "$PI05_ROOT/.venv/bin/python" ]]; then
    echo 'Missing .venv. Run: bash deploy/fr3_wuji/setup.sh' >&2
    exit 1
fi
# Explicit GPU selection prevents silent CPU fallback in model serving.
export JAX_PLATFORMS=cuda
exec taskset -c "$PI05_CPUSET" nice -n 10 "$PI05_ROOT/.venv/bin/python" "$@"
