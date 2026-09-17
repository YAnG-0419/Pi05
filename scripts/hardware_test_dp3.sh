#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-help}"
checkpoint="${DP3_CHECKPOINT:-$root/models/latest.ckpt}"
uri="${DP3_URI:-ws://127.0.0.1:8001}"
if [[ "$mode" != help && ! -f "$checkpoint" ]]; then
    printf 'DP3 checkpoint not found: %s\n' "$checkpoint" >&2
    exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  check)
    exec bash "$root/deploy.sh" dp3 --check --checkpoint "$checkpoint" --uri "$uri" "$@" ;;
  read-only)
    exec bash "$root/deploy.sh" dp3 --read-only --checkpoint "$checkpoint" --uri "$uri" "$@" ;;
  supervised-trial)
    if [[ "${DP3_SUPERVISED_TRIAL_ACK:-}" != "I_HAVE_ESTOP_AND_OPERATOR" ]]; then
      echo 'Set DP3_SUPERVISED_TRIAL_ACK=I_HAVE_ESTOP_AND_OPERATOR after clearing the workspace.' >&2
      exit 2
    fi
    exec bash "$root/deploy.sh" dp3 --execute --supervised-trial --start-cameras \
      --minimum-time-scale "${DP3_TRIAL_TIME_SCALE:-10}" --finish-policy disable \
      --checkpoint "$checkpoint" --uri "$uri" "$@" ;;
  help|-h|--help)
    printf '%s\n' 'Usage: DP3_CHECKPOINT=/path/latest.ckpt [DP3_URI=ws://host:8001] bash scripts/hardware_test_dp3.sh {check|read-only|supervised-trial} [deployment args...]' ;;
  *) printf 'Unknown test stage: %s\n' "$mode" >&2; exit 2 ;;
esac
