"""Resume this machine's prepared dependency downloads; never installs packages or starts hardware."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WHEELS = ROOT / ".deployment/wheels"


def matches(path: Path, digest: str) -> bool:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == digest.removeprefix("sha256:")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Show remaining files without network access.")
    parser.add_argument("--verify-only", action="store_true", help="Require all files and verify their SHA256 hashes.")
    args = parser.parse_args()
    manifest_path = ROOT / ".deployment/dependency-downloads.json"
    manifest = json.loads(manifest_path.read_text())
    if not matches(ROOT / "uv.lock", manifest["lock_sha256"]):
        parser.error("uv.lock changed; regenerate the download manifest before continuing")
    WHEELS.mkdir(parents=True, exist_ok=True)
    # A second invocation must not write the same partial files concurrently.
    with (WHEELS.parent / "download.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another dependency download/verification is running")
        remaining = []
        for wheel in manifest["wheels"]:
            path = WHEELS / wheel["filename"]
            if path.exists():
                if not matches(path, wheel["hash"]):
                    parser.error(f"SHA256 mismatch: {path}; move that file aside before retrying")
                continue
            remaining.append(wheel)
        print(f"Verified: {len(manifest['wheels']) - len(remaining)}/{len(manifest['wheels'])}", flush=True)
        for index, wheel in enumerate(remaining, 1):
            path = WHEELS / wheel["filename"]
            partial = path.with_suffix(".whl.partial")
            downloaded = partial.stat().st_size / 1024**2 if partial.exists() else 0
            total = f"{wheel['size'] / 1024**2:.1f}" if "size" in wheel else "~991.2"
            print(f"[{index}/{len(remaining)}] {path.name}: {downloaded:.1f}/{total} MiB", flush=True)
            if args.status or args.verify_only:
                continue
            subprocess.run(
                [
                    "curl",
                    "--http1.1",
                    "--fail",
                    "--location",
                    "--progress-bar",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "15",
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    wheel["url"],
                ],
                check=True,
            )
            if not matches(partial, wheel["hash"]):
                parser.error(f"SHA256 mismatch: {partial}; move that partial file aside before retrying")
            partial.rename(path)
            print("SHA256 OK", flush=True)
        if args.verify_only and remaining:
            raise SystemExit("Some dependencies are still missing; run without --verify-only to resume downloading.")
        if not args.status:
            print("All prepared dependency wheels verified. Next: bash deploy/fr3_wuji/finish_setup.sh")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nStopped; partial downloads are preserved. Run the same command to resume.") from None
