#!/usr/bin/env bash
set -euo pipefail
DP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$DP_ROOT"
export PYTHONDONTWRITEBYTECODE=1
source "$DP_ROOT/scripts/python_env.sh"
exec "$DP_PYTHON" -m "dp_deploy.${1:-doctor}" "${@:2}"
