#!/usr/bin/env bash
# Sourced by workspace entrypoints. DP_PYTHON may name a path or a PATH command.
if [[ -n "${DP_PYTHON:-}" ]]; then
  if ! DP_PYTHON="$(command -v -- "$DP_PYTHON")"; then
    echo "DP_PYTHON 不存在，请指定部署机 Python 的实际路径。" >&2
    return 1
  fi
elif [[ -x "$DP_ROOT/.venv/bin/python" ]]; then
  DP_PYTHON="$DP_ROOT/.venv/bin/python"
else
  echo "缺少项目 Python 环境。请运行 bash scripts/setup_env.sh，或设置 DP_PYTHON 指向部署机现有模型环境。" >&2
  return 1
fi
export DP_PYTHON
