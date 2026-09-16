"""Offline deployment checks. Never constructs ROS nodes, SDK hands, or robot clients."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from .config import ROOT, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--container", action="store_true")
    p.add_argument("--checkpoint", action="store_true")
    a = p.parse_args()
    c = load_config()
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, fn):
        try:
            result = fn()
            checks.append({"check": name, "ok": True, "detail": result})
        except Exception as e:
            checks.append({"check": name, "ok": False, "detail": str(e)})

    from .hardware import limits, verify_overlay, docker

    def model_environment():
        import torch
        import torchvision
        import diffusers
        import av
        import zarr

        if not torch.cuda.is_available():
            raise RuntimeError("GPU/CUDA unavailable on deployment host")
        return {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "diffusers": diffusers.__version__,
            "av": av.__version__,
            "zarr": zarr.__version__,
            "gpu": torch.cuda.get_device_name(0),
        }

    check("model_environment", model_environment)
    check("reference_limits", lambda: len(limits(c)[0]["lower"]))
    check("controller_overlay", lambda: verify_overlay()["built_library_sha256"])
    check("checkpoint_exists", lambda: Path(c["checkpoint"]).stat().st_size)
    check("sdk", lambda: str(next((Path(c["sdk_site"]) / "wuji_sdk").glob("*.so"))))
    check(
        "container_image",
        lambda: subprocess.check_output(
            ["docker", "image", "inspect", c["docker_image"], "--format", "{{.Id}}"],
            text=True,
        ).strip(),
    )

    def cpus():
        import os

        available = os.sched_getaffinity(0)
        sets = []
        for spec in [*c["controller_cpus"].values(), c["housekeeping_cpus"]]:
            ids = set()
            for part in spec.split(","):
                ends = list(map(int, part.split("-")))
                ids.update(range(ends[0], ends[-1] + 1))
            if not ids <= available:
                raise ValueError("Unavailable CPUs " + spec)
            sets.append(ids)
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Overlapping CPU sets")
        return "available and disjoint; realtime scheduling still requires field verification"

    check("cpu_sets", cpus)
    if a.checkpoint:
        from .checkpoint import read_checkpoint, digest

        def model():
            cfg, states = read_checkpoint(c["checkpoint"])
            sha = digest(c["checkpoint"])
            expected = (
                (Path(c["checkpoint"]).parent / "SHA256SUMS.txt").read_text().split()[0]
            )
            if sha != expected:
                raise ValueError("Checkpoint checksum differs from SHA256SUMS.txt")
            (ROOT / "reports/checkpoint-config.json").write_text(
                json.dumps(cfg, indent=2) + "\n"
            )
            return {
                "sha256": sha,
                "weights": list(states),
                "horizon": cfg["horizon"],
                "n_obs_steps": cfg["n_obs_steps"],
                "n_action_steps": cfg["n_action_steps"],
            }

        check("checkpoint_config_checksum", model)
    if a.container:

        def imports():
            out = ROOT / "reports/container-check"
            out.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="dp-import-") as ipc:
                cmd = docker(
                    c,
                    "dp-import-check",
                    out,
                    Path(ipc),
                    [
                        "python3",
                        "-c",
                        'import rclpy, sensor_msgs, teleop_interfaces, teleop_core, wuji_sdk; from dp_deploy.robot import bridge; print("ROS messages / SDK / bridge imports OK")',
                    ],
                    sdk=True,
                )
                cmd[cmd.index("--network") + 1] = "none"
                return subprocess.check_output(cmd, text=True).strip()

        check("isolated_container_imports", imports)
    report = {
        "hardware_output": False,
        "hardware_connected": False,
        "checks": checks,
        "all_passed": all(x["ok"] for x in checks),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/doctor.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
