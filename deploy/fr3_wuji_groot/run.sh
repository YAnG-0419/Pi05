#!/usr/bin/env bash
set -euo pipefail
GROOT_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$GROOT_DEPLOY_DIR/../fr3_wuji/env.sh"
export PYTHONDONTWRITEBYTECODE=1
exec taskset -c "$PI05_CPUSET" "$PI05_ROOT/.venv/bin/python" -B \
  -m deploy.fr3_wuji_groot.deploy "$@"
