#!/usr/bin/env bash
# Install the prepared wheels into this project and run software-only checks.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
python3 deploy/fr3_wuji/download_deps.py --verify-only
uv_bin="$PI05_ROOT/.deployment/tools/bin/uv"
mapfile -t wheel_files < <(python3 -c 'import json; from pathlib import Path; root = Path.cwd(); manifest = json.loads((root / ".deployment/dependency-downloads.json").read_text()); print("\n".join(str(root / ".deployment/wheels" / wheel["filename"]) for wheel in manifest["wheels"]))')
taskset -c "$PI05_CPUSET" nice -n 10 "$uv_bin" pip install \
    --python "$PI05_ROOT/.venv/bin/python" --offline --no-deps "${wheel_files[@]}"
taskset -c "$PI05_CPUSET" nice -n 10 "$uv_bin" sync --locked --offline --python 3.11
"$uv_bin" pip check --python "$PI05_ROOT/.venv/bin/python"
exec bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/doctor.py --prepare-tokenizer
