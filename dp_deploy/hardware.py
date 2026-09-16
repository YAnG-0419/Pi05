"""Local launcher for the inherited pi0.5 ROS/SDK boundary, without model assumptions."""

import ast
import json
import os
from pathlib import Path
import subprocess
import uuid
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from .config import ROOT, load_config
from .contract import joint_names
from .robot.qualification import digest

OVERLAY = ROOT / ".deployment/overlay"


def limits(config):
    ref = Path(config["reference"])
    safety = ref / "ros_ws/src/teleop_core/teleop_core/safety.py"
    found = {}
    for node in ast.parse(safety.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("LOWER_LIMITS", "UPPER_LIMITS"):
                    expr = node.value.args[0]
                    found[t.id] = np.asarray(
                        ast.literal_eval(expr.left) * ast.literal_eval(expr.right)
                    )
    lo = np.empty(54)
    hi = np.empty(54)
    paths = [safety]
    arm = np.r_[0:7, 27:34]
    lo[arm] = found["LOWER_LIMITS"]
    hi[arm] = found["UPPER_LIMITS"]
    for side, offset in [("left", 7), ("right", 34)]:
        p = ref / f"adapters/wuji/config/retarget_manus_wuji_hand_2_{side}.yaml"
        urdf = (
            p.parent / yaml.safe_load(p.read_text())["optimizer"]["urdf_path"]
        ).resolve()
        paths.extend([p, urdf])
        joints = {
            j.attrib["name"]: j.find("limit")
            for j in ET.parse(urdf).getroot().findall("joint")
        }
        for i, name in enumerate(joint_names(side + "_hand"), offset):
            lo[i] = float(joints[name].attrib["lower"])
            hi[i] = float(joints[name].attrib["upper"])
    if not np.isfinite([lo, hi]).all() or not np.all(lo < hi):
        raise ValueError("Invalid hardware limits")
    return {"lower": lo.tolist(), "upper": hi.tolist()}, {
        str(p): digest(p) for p in paths
    }


def verify_overlay():
    m = json.loads((OVERLAY / "source-manifest.json").read_text())
    expected_library = (
        OVERLAY
        / "install/franka_fr3_arm_controllers/lib/libfranka_fr3_arm_controllers.so"
    )
    expected_reference = (
        Path(load_config()["reference"]) / "ros_ws/src/franka_fr3_arm_controllers"
    )
    if (
        Path(m.get("built_library", "")) != expected_library
        or Path(m["reference"]) != expected_reference
    ):
        raise ValueError(
            "Overlay belongs to another location; run ./run.sh build --fresh on this host"
        )
    if m["build_completed"] is not True:
        raise ValueError("Run ./run.sh build first")
    for root, key in [
        (Path(m["reference"]), "source_hashes"),
        (OVERLAY / "src/franka_fr3_arm_controllers", "overlay_source_hashes"),
    ]:
        for name, sha in m[key].items():
            if digest(root / name) != sha:
                raise ValueError("Controller source changed: " + name)
    if digest(m["built_library"]) != m["built_library_sha256"]:
        raise ValueError("Controller library changed")
    return m


def runtime(c, output, execute):
    bounds, hashes = limits(c)
    m = verify_overlay() if execute else None
    workcell = yaml.safe_load(
        (Path(c["reference"]) / "config/workcell/current.yaml").read_text()
    )
    for side in ("left", "right"):
        workcell[side.upper()]["robot_ip"] = c[side + "_arm_ip"]
        workcell[side.upper()]["controller_cpus"] = c["controller_cpus"][side]
        if str(workcell[side.upper()]["use_fake_hardware"]).lower() != "false":
            raise ValueError("Expected real workcell")
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    r = {
        "reference": c["reference"],
        "hands": {s: c[s + "_hand_address"] for s in ("left", "right")},
        "arms": {s: c[s + "_arm_ip"] for s in ("left", "right")},
        "limits": bounds,
        "reference_hashes": hashes,
        "controller_library": m["built_library"] if m else "",
        "controller_sha256": m["built_library_sha256"] if m else "",
        "hand_control": "slider",
        "finish_policy": "disable",
        "continuous": False,
        "arm_speed_rad_s": c["arm_speed_rad_s"],
    }
    (output / "runtime.json").write_text(json.dumps(r, indent=2) + "\n")
    return r


def docker(c, name, output, ipc, command, controller=False, sdk=False):
    # Absolute mount paths preserve colcon prefixes and source hashes.
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--cpuset-cpus",
        ",".join(c["controller_cpus"].values())
        if controller
        else c["housekeeping_cpus"],
        "-v",
        f"{ROOT}:{ROOT}:ro",
        "-v",
        f"{output}:{output}",
        "-v",
        f"{ipc}:{ipc}",
        "-v",
        f"{c['reference']}:{c['reference']}:ro",
        "-w",
        str(ROOT),
    ]
    for v in (
        "PYTHONDONTWRITEBYTECODE=1",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        f"ROS_LOCALHOST_ONLY={c['ros_localhost_only']}",
        f"ROS_DOMAIN_ID={c['ros_domain_id']}",
        f"RMW_IMPLEMENTATION={c['rmw_implementation']}",
        f"ROS_LOG_DIR={output}/ros",
        f"PYTHONPATH={ROOT}" + (":/sdk" if sdk else ""),
        "HOME=/tmp",
        f"DP_DEPLOY_CONFIG={output}/deploy-config.json",
    ):
        cmd += ["-e", v]
    if controller:
        cmd += [
            "--privileged",
            "--ulimit",
            "rtprio=99",
            "--ulimit",
            "rttime=-1",
            "--ulimit",
            "memlock=-1",
        ]
    else:
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    if sdk:
        for package in ("wuji_sdk", "wuji_sdk.libs"):
            path = Path(c["sdk_site"]) / package
            if not path.is_dir():
                raise ValueError("Missing SDK: " + str(path))
            cmd += ["-v", f"{path}:/sdk/{package}:ro"]
    setup = OVERLAY / "install/setup.bash"
    # Positional shell arguments avoid interpolating paths into shell code.
    return cmd + [
        c["docker_image"],
        "bash",
        "-c",
        'source "$1"; shift; exec "$@"',
        "dp",
        str(setup),
        *command,
    ]


def commands(c, output, ipc, execute, bridge=True):
    prefix = "dp-" + uuid.uuid4().hex[:8]
    result = {}
    if execute:
        result["arms"] = docker(
            c,
            prefix + "-arms",
            output,
            ipc,
            [
                "ros2",
                "launch",
                "franka_fr3_arm_controllers",
                "franka_fr3_arm_controllers.launch.py",
                f"robot_config_file:={output}/workcell.yaml",
            ],
            controller=True,
        )
        for role in ("gateway", "splitter"):
            result[role] = docker(
                c,
                prefix + "-" + role,
                output,
                ipc,
                [
                    "python3",
                    "-m",
                    "dp_deploy.robot.ros_boundary",
                    role,
                    "--reference",
                    c["reference"],
                    "--arm-speed-rad-s",
                    str(c["arm_speed_rad_s"]),
                ],
            )
    if bridge:
        command = [
            "python3",
            "-m",
            "dp_deploy.robot.bridge",
            "--runtime",
            str(output / "runtime.json"),
            "--socket",
            str(ipc / "devices.sock"),
        ]
        if execute:
            command += ["--execute", "--supervised-trial"]
        result["devices"] = docker(
            c, prefix + "-devices", output, ipc, command, sdk=True
        )
    from urllib.parse import urlparse

    port = urlparse(c["observation_url"]).port
    result["observations"] = docker(
        c,
        prefix + "-obs",
        output,
        ipc,
        ["python3", "-m", "dp_deploy.observe", "--port", str(port)],
    )
    return result


def preflight_owners(execute):
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    containers = subprocess.check_output(
        ["docker", "ps", "--no-trunc", "--format", "{{.Names}} {{.Command}}"], text=True
    )
    markers = (
        "apps.operator_gui",
        "apps.wuji_ui",
        "adapters.wuji.hand_only",
        "hand_read.py",
        "observation_sources.py hands",
    )
    if any(s in processes for s in markers):
        raise RuntimeError("Existing hand SDK owner; stop it before deployment")
    markers = (
        "oneshot.bridge",
        "dp_deploy.robot.bridge",
        "hand-control",
        "adapters.wuji",
    )
    if any(s in containers for s in markers):
        raise RuntimeError("Existing hand device bridge")
    if execute and (
        any(
            s in containers
            for s in (
                "franka-control",
                "robot_control.launch",
                "franka_fr3_arm_controllers.launch",
                "moveit-real",
            )
        )
        or "read_arms" in processes
    ):
        raise RuntimeError("Existing arm owner; refusing a second FCI connection")


class Children:
    def __init__(self, output):
        self.output = output
        self.children = []
        self.files = []

    def launch(self, label, command):
        f = (self.output / (label + ".log")).open("x")
        self.files.append(f)
        self.children.append(
            (
                command[command.index("--name") + 1],
                subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT),
            )
        )

    def check(self):
        if any(p.poll() is not None for _, p in self.children):
            raise RuntimeError("A hardware/observation process exited; inspect logs")

    def close(self):
        for name, p in reversed(self.children):
            subprocess.run(
                ["docker", "stop", "--signal", "SIGINT", "--timeout", "10", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.terminate()
                p.wait(timeout=5)
        for f in self.files:
            f.close()
