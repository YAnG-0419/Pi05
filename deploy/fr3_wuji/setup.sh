#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
uv_bin="$PI05_ROOT/.deployment/tools/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
    echo 'Missing project-local uv. See deploy/fr3_wuji/README.md for bootstrap.' >&2
    exit 1
fi
exec taskset -c "$PI05_CPUSET" nice -n 10 "$uv_bin" sync --locked --python 3.11
