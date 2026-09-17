#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${DP3_ENV_NAME:-dp3-deploy}"
if conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
    conda env update -n "$env_name" -f "$root/environment_dp3.yml"
else
    conda env create -n "$env_name" -f "$root/environment_dp3.yml"
fi
conda run -n "$env_name" python -m pip install \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
bash "$root/fr3_wuji/setup.sh"
echo "DP3 model environment and hardware client environment are ready: $env_name / $root/.venv"
