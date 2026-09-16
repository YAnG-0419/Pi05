#!/usr/bin/env bash
set -euo pipefail
DP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DP_BASE_PYTHON="${1:-${DP_BASE_PYTHON:-python3.10}}"
command -v -- "$DP_BASE_PYTHON" >/dev/null || {
  echo "找不到 $DP_BASE_PYTHON；请安装 Python 3.10 和 venv，或把 Python 3.10 路径作为第一个参数。" >&2
  exit 1
}
"$DP_BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for this pinned deployment environment"'
"$DP_BASE_PYTHON" -m venv "$DP_ROOT/.venv"
"$DP_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$DP_ROOT/.venv/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r "$DP_ROOT/requirements-model.txt"
"$DP_ROOT/.venv/bin/python" -m pip check
"$DP_ROOT/.venv/bin/python" -c 'import torch, torchvision, diffusers, av, zarr; print("torch", torch.__version__); assert torch.cuda.is_available(), "GPU/CUDA unavailable: check NVIDIA driver"; print(torch.cuda.get_device_name(0))'
echo "模型环境就绪：$DP_ROOT/.venv/bin/python"
