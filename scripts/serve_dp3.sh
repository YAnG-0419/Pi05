#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${DP3_CHECKPOINT:-$root/models/latest.ckpt}"
export PYTHONPATH="$root:$root/src:$root/packages/openpi-client/src:$root/fr3_wuji"
exec conda run --no-capture-output -n "${DP3_ENV_NAME:-dp3-deploy}" \
    python "$root/deployment/dp3_server.py" --checkpoint "$checkpoint" \
    --host "${DP3_HOST:-127.0.0.1}" --port "${DP3_PORT:-8001}" "$@"
