#!/usr/bin/env bash
set -euo pipefail
EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Reuse the existing process-local launcher. Never install dependencies, edit
# configuration, or apply system CPU/IRQ/driver changes.
export PYTHONDONTWRITEBYTECODE=1
exec bash "$EVAL_ROOT/deploy/fr3_wuji/run.sh" "$EVAL_ROOT/experiments/weight_motion_eval/cli.py" "$@"
