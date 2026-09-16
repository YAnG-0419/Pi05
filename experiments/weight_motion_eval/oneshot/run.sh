#!/usr/bin/env bash
set -euo pipefail
PI05_ONESHOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PI05_ONESHOT_DIR/../../../deploy/fr3_wuji/env.sh"
exec taskset -c "$PI05_CPUSET" "$PI05_ROOT/.venv/bin/python" -m experiments.weight_motion_eval.oneshot.deploy "$@"
