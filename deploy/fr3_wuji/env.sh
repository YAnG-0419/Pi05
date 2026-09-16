#!/usr/bin/env bash
# Source only from deployment wrappers; environment changes stay in their process.
PI05_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="$PI05_ROOT/.venv"
export UV_CACHE_DIR="$PI05_ROOT/.deployment/cache/uv"
export UV_PYTHON_INSTALL_DIR="$PI05_ROOT/.deployment/python"
export UV_PYTHON_BIN_DIR="$PI05_ROOT/.deployment/bin"
export XDG_CACHE_HOME="$PI05_ROOT/.deployment/cache"
export HF_HOME="$PI05_ROOT/.deployment/cache/huggingface"
export OPENPI_DATA_HOME="$PI05_ROOT/.deployment/cache/openpi"
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${PI05_CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# This workcell reserves CPUs 8-15 for Franka FCI. Match its housekeeping set.
export PI05_CPUSET="${PI05_CPUSET:-0-7,16-23}"
# Prevent activated Conda/ROS shells from injecting their Python or CUDA libraries.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH
export PATH="$PI05_ROOT/.venv/bin:/usr/local/bin:/usr/bin:/bin"
cd -- "$PI05_ROOT"
