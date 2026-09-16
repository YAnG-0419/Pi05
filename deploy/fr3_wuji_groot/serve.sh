#!/usr/bin/env bash
set -euo pipefail
GROOT_DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="$(cd -- "$GROOT_DEPLOY_DIR/../.." && pwd)"
GROOT_ROOT="${GROOT_ROOT:-/home/descfly/workspace/Isaac-GR00T}"
GROOT_MODEL_CACHE="${GROOT_MODEL_CACHE:-$GROOT_ROOT/outputs/dual_franka_wuji/models_http}"
if [[ ! -x "$GROOT_ROOT/.venv/bin/python" ]]; then
  echo "Missing Isaac-GR00T Python environment: $GROOT_ROOT/.venv/bin/python" >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PI05_ROOT:$GROOT_ROOT"
export OMP_NUM_THREADS="${GROOT_OMP_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
cd -- "$PI05_ROOT"
exec "$GROOT_ROOT/.venv/bin/python" -B -m deploy.fr3_wuji_groot.model_server \
  --model-cache "$GROOT_MODEL_CACHE" "$@"
