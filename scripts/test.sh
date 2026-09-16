#!/usr/bin/env bash
set -euo pipefail
DP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DP_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$DP_ROOT/.testdeps${PYTHONPATH:+:$PYTHONPATH}"
source "$DP_ROOT/scripts/python_env.sh"
mkdir -p reports
exec "$DP_PYTHON" -m pytest -q tests --junitxml=reports/tests.xml "$@"
