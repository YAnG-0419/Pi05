#!/usr/bin/env bash
set -euo pipefail
PI05_ONESHOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PI05_OVERLAY="$PI05_ONESHOT_ROOT/.deployment/oneshot-overlay"
test -f "$PI05_OVERLAY/source-manifest.json"
docker run --rm --network none --cpuset-cpus 0-7,16-23 \
  --user "$(id -u):$(id -g)" \
  -v /home/user/lpy/gello-retarget:/workspace/franka_upper_body_teleop:ro \
  -v "$PI05_OVERLAY":/overlay -w /overlay \
  -e PYTHONDONTWRITEBYTECODE=1 \
  franka-upper-body-teleop:latest bash -c \
  'colcon build --packages-select franka_fr3_arm_controllers --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release --parallel-workers 1'
cd -- "$PI05_ONESHOT_ROOT"
"$PI05_ONESHOT_ROOT/.venv/bin/python" -m experiments.weight_motion_eval.oneshot.record_build
