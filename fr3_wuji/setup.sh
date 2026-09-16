#!/usr/bin/env bash
set -euo pipefail
# Locate uv before env.sh removes unrelated environment paths.
PI05_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI05_UV="${OPENPI_UV:-$(command -v uv || true)}"
if [[ -z "$PI05_UV" ]]; then
    echo 'Install uv or set OPENPI_UV to its executable path.' >&2
    exit 1
fi
source "$PI05_SETUP_DIR/env.sh"
exec taskset -c "$PI05_CPUSET" nice -n 10 "$PI05_UV" sync --locked --python 3.11
