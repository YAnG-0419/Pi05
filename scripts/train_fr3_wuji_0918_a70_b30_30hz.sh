#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/train.py" pi05_fr3_wuji_0918_a70_b30_30hz \
  --exp-name tomato_lora_0918_a70_b30_30hz "$@"
