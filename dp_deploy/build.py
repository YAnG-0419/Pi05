"""Build a fresh host-specific controller overlay without connecting to devices."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid
from .config import load_config, ROOT


def prepare_build(output, reference, fresh=False):
    """Keep prior artifacts as a backup; never reuse a relocated CMake cache."""
    manifest = output / "source-manifest.json"
    if manifest.exists() and not fresh:
        m = json.loads(manifest.read_text())
        expected = (
            output
            / "install/franka_fr3_arm_controllers/lib/libfranka_fr3_arm_controllers.so"
        )
        old_library = m.get("built_library")
        if Path(m["reference"]) != reference or (
            old_library is not None and Path(old_library) != expected
        ):
            raise ValueError(
                "Relocated overlay detected. Run ./run.sh build --fresh on the deployment host."
            )
    if fresh and output.exists():
        backup = output.with_name(
            "overlay-backup-"
            + time.strftime("%Y%m%d-%H%M%S")
            + "-"
            + uuid.uuid4().hex[:6]
        )
        output.rename(backup)
        print("Previous overlay saved:", backup)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Archive old overlay and rebuild from configured reference source",
    )
    a = p.parse_args()
    c = load_config()
    from .robot.prepare_overlay import prepare, OUTPUT, REFERENCE

    for name in ("franka_fr3_arm_controllers", "teleop_interfaces", "teleop_core"):
        if not (Path(c["reference"]) / "ros_ws/src" / name).is_dir():
            raise FileNotFoundError(
                f"Missing reference ROS package {name}: {c['reference']}"
            )
    subprocess.run(
        ["docker", "image", "inspect", c["docker_image"]],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".deployment/owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_build(OUTPUT, REFERENCE, fresh=a.fresh)
        if not (OUTPUT / "source-manifest.json").exists():
            prepare()
        for name in ("teleop_interfaces", "teleop_core"):
            target = OUTPUT / "src" / name
            if not target.exists():
                shutil.copytree(
                    Path(c["reference"]) / "ros_ws/src" / name,
                    target,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-e",
                "HOME=/tmp",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-v",
                f"{OUTPUT}:{OUTPUT}",
                "-w",
                str(OUTPUT),
                c["docker_image"],
                "bash",
                "-c",
                "colcon build --packages-select teleop_interfaces teleop_core franka_fr3_arm_controllers "
                "--cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release --parallel-workers 2",
            ],
            check=True,
        )
        from .robot.record_build import main as record

        record()
    print("Built without hardware:", OUTPUT)


if __name__ == "__main__":
    main()
