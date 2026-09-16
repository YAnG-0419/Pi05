#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="$(cd -- "$DEPLOY_DIR/../.." && pwd)"
GROOT_ROOT="${GROOT_ROOT:-$(cd -- "$PI05_ROOT/../Isaac-GR00T" && pwd)}"
GROOT_IMAGE="${GROOT_IMAGE:-pi05-groot-server:scratch30k-20260916}"
GROOT_BASE_IMAGE="${GROOT_BASE_IMAGE:-gr00t:latest}"

if [[ ! -f "$GROOT_ROOT/pyproject.toml" || ! -d "$GROOT_ROOT/.git" ]]; then
  echo "Invalid GROOT_ROOT: $GROOT_ROOT" >&2
  exit 1
fi
if ! docker image inspect "$GROOT_BASE_IMAGE" >/dev/null 2>&1; then
  echo "Missing base image $GROOT_BASE_IMAGE" >&2
  echo "Build it first: cd '$GROOT_ROOT' && bash docker/build.sh" >&2
  exit 1
fi
if [[ -n "$(git -C "$GROOT_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "Isaac-GR00T must be clean because its Docker source is archived from HEAD" >&2
  git -C "$GROOT_ROOT" status --short >&2
  exit 1
fi

PI05_REVISION="$(git -C "$PI05_ROOT" rev-parse HEAD)"
GROOT_REVISION="$(git -C "$GROOT_ROOT" rev-parse HEAD)"
if [[ -n "$(git -C "$PI05_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  PI05_REVISION="${PI05_REVISION}-dirty"
fi
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GROOT_CONTEXT="$(mktemp -d /tmp/pi05-groot-source.XXXXXX)"
trap 'rm -rf -- "$GROOT_CONTEXT"' EXIT

# A Git archive prevents ignored 20+ GB outputs and local credentials from
# entering the Docker build context. Only the selected committed source is used.
git -C "$GROOT_ROOT" archive --format=tar HEAD | tar -xf - -C "$GROOT_CONTEXT"

docker buildx build --load \
  --build-context "groot_src=$GROOT_CONTEXT" \
  --build-arg "GROOT_BASE_IMAGE=$GROOT_BASE_IMAGE" \
  --build-arg "PI05_REVISION=$PI05_REVISION" \
  --build-arg "GROOT_REVISION=$GROOT_REVISION" \
  --build-arg "BUILD_DATE=$BUILD_DATE" \
  --file "$DEPLOY_DIR/Dockerfile.inference" \
  --tag "$GROOT_IMAGE" \
  "$@" \
  "$PI05_ROOT"

echo "Built $GROOT_IMAGE"
docker image inspect "$GROOT_IMAGE" --format '{{json .Config.Labels}}'
