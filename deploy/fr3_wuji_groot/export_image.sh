#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 IMAGE OUTPUT.tar" >&2
  exit 2
fi
IMAGE="$1"
ARCHIVE="$2"
ARCHIVE_DIR="$(cd -- "$(dirname -- "$ARCHIVE")" && pwd)"
ARCHIVE_NAME="$(basename -- "$ARCHIVE")"
ARCHIVE="$ARCHIVE_DIR/$ARCHIVE_NAME"
CHECKSUM="$ARCHIVE_DIR/${ARCHIVE_NAME}.sha256"

if [[ -e "$ARCHIVE" || -e "$CHECKSUM" ]]; then
  echo "Refusing to overwrite: $ARCHIVE or $CHECKSUM already exists" >&2
  exit 1
fi
if [[ ! -d "$(dirname -- "$ARCHIVE")" ]]; then
  echo "Output directory does not exist: $(dirname -- "$ARCHIVE")" >&2
  exit 1
fi
docker image inspect "$IMAGE" >/dev/null
docker save --output "$ARCHIVE" "$IMAGE"
(cd -- "$ARCHIVE_DIR" && sha256sum "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.sha256")
echo "Exported $ARCHIVE"
echo "Checksum $CHECKSUM"
