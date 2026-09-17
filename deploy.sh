#!/usr/bin/env bash
# Two execution strategies with independent pi05 model selection. Hardware requires --execute.
set -euo pipefail
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$DEPLOY_MODE" in
  dp3) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.dp3 "$@" ;;
  serve-dp3) exec bash "$DEPLOY_ROOT/scripts/serve_dp3.sh" "$@" ;;
  check) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" "$DEPLOY_ROOT/scripts/check_deployment.py" "$@" ;;
  setup) exec bash "$DEPLOY_ROOT/fr3_wuji/setup.sh" "$@" ;;
  serve)
    DEPLOY_MODEL="${1:-19999}"
    if [[ $# -gt 0 ]]; then shift; fi
    case "$DEPLOY_MODEL" in
      19999|30000) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.server --model "$DEPLOY_MODEL" "$@" ;;
      slow54|slow64|normal) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.server --profile "$DEPLOY_MODEL" "$@" ;;
      -*) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.server "$DEPLOY_MODEL" "$@" ;;
      *) printf 'Unknown model: %s\n' "$DEPLOY_MODEL" >&2; exit 2 ;;
    esac ;;
  slow54|slow64)
    exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.run --profile "$DEPLOY_MODE" "$@" ;;
  serve-19999) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.server --profile slow54 "$@" ;;
  serve-30000) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.server --profile slow64 "$@" ;;
  slow|normal) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.run --strategy "$DEPLOY_MODE" "$@" ;;
  fast) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.run --profile normal "$@" ;;
  30000) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m deployment.run --profile slow64 "$@" ;;
  prepare-controller) exec bash "$DEPLOY_ROOT/fr3_wuji/run.sh" -m experiments.weight_motion_eval.oneshot.prepare_overlay "$@" ;;
  build-controller) exec bash "$DEPLOY_ROOT/experiments/weight_motion_eval/oneshot/build_overlay.sh" "$@" ;;
  help|-h|--help) printf '%s\n' 'Usage: DP3_CHECKPOINT=/path/model.ckpt bash deploy.sh serve-dp3
       bash deploy.sh dp3 {--check|--read-only|--execute} --checkpoint /path/model.ckpt [args...]
       bash deploy.sh serve <19999|30000> [args...]
       bash deploy.sh <slow|normal> --model <19999|30000> [args...]
       bash deploy.sh {check|setup|prepare-controller|build-controller}' 'Legacy aliases: slow54, slow64, fast, 30000, serve-19999, serve-30000' ;;
  *) printf 'Unknown deployment mode: %s\n' "$DEPLOY_MODE" >&2; exit 2 ;;
esac
