"""Render deployment, inspect feedback, or execute one fresh 50-step prediction.

Default --check only writes a reviewable launch configuration. Physical output
requires --execute plus measured commissioning evidence or --supervised-trial.
"""

import argparse
from contextlib import ExitStack
import copy
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

import yaml

from .ipc import RemoteDevices
from .limits import HAND_SPEED_RAD_S
from .policy_server import checkpoint_contract
from .qualification import Qualification
from .qualification import SupervisedTrial
from .qualification import digest

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path("/home/user/lpy/gello-retarget")
SDK = Path("/home/user/miniconda3/envs/gello-upper-body-teleop/lib/python3.10/site-packages")
OVERLAY = ROOT / ".deployment/oneshot-overlay"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--read-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--left-arm-ip", default="172.16.0.2")
    parser.add_argument("--right-arm-ip", default="172.16.1.2")
    parser.add_argument("--wuji-sides", choices=("both",), default="both")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:7447")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/19999")
    parser.add_argument("--uri", default="ws://127.0.0.1:8001")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--finish-policy", choices=("hold", "disable"), default="hold")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--supervised-trial",
        action="store_true",
        help="One operator-supervised 50-step run without historical commissioning evidence",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/weight_motion_eval" / ("deployment-" + uuid.uuid4().hex[:10])
    )
    args = parser.parse_args(argv)
    if args.supervised_trial and (not args.execute or args.qualification is not None):
        parser.error("--supervised-trial requires --execute and cannot be combined with --qualification")
    for value in (args.left_arm_ip, args.right_arm_ip):
        ipaddress.IPv4Address(value)
    for value in (args.wuji_left_address, args.wuji_right_address):
        address, port = value.rsplit(":", 1)
        ipaddress.IPv4Address(address)
        if not 1 <= int(port) <= 65535:
            parser.error("Invalid hand port")
    if args.left_arm_ip == args.right_arm_ip or args.wuji_left_address == args.wuji_right_address:
        parser.error("Left/right devices must have distinct addresses")
    return args


def verify_overlay():
    manifest = json.loads((OVERLAY / "source-manifest.json").read_text())
    if manifest.get("build_completed") is not True:
        raise ValueError("Build the isolated controller overlay first")
    for root, key in (
        (Path(manifest["reference"]), "source_hashes"),
        (OVERLAY / "src/franka_fr3_arm_controllers", "overlay_source_hashes"),
    ):
        for name, expected in manifest[key].items():
            if digest(root / name) != expected:
                raise ValueError("Controller source changed since build: " + str(root / name))
    if digest(manifest["built_library"]) != manifest["built_library_sha256"]:
        raise ValueError("Installed controller differs from recorded build")
    return manifest


def write_runtime(args, output):
    from experiments.weight_motion_eval.reference import reference_limits

    limits, names = reference_limits()
    manifest = verify_overlay()
    model, _ = checkpoint_contract(args.checkpoint)
    workcell = yaml.safe_load((REFERENCE / "config/workcell/current.yaml").read_text())
    workcell["LEFT"]["robot_ip"], workcell["RIGHT"]["robot_ip"] = args.left_arm_ip, args.right_arm_ip
    for side in ("LEFT", "RIGHT"):
        if workcell[side]["use_fake_hardware"] != "false":
            raise ValueError("Expected the reference real workcell profile")
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    runtime = {
        "schema_version": 1,
        "reference": str(REFERENCE),
        "hands": {"left": args.wuji_left_address, "right": args.wuji_right_address},
        "arms": {"left": args.left_arm_ip, "right": args.right_arm_ip},
        "limits": {k: limits[k].tolist() for k in ("lower", "upper")},
        "reference_hashes": limits["sources"],
        "joint_names": names,
        "controller_library": manifest["built_library"],
        "controller_sha256": manifest["built_library_sha256"],
        "checkpoint": model,
        "finish_policy": args.finish_policy,
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime, limits, names


def docker_command(name, output, ipc, command, *, controller=False, sdk=False):
    # Match reference compose: DDS only on loopback; keep SDK/ROS helpers off FCI CPUs.
    args = [
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
        "8-9,12-13" if controller else "0-7,16-23",
        "-v",
        f"{REFERENCE}:/workspace/franka_upper_body_teleop:ro",
        "-v",
        f"{REFERENCE}:{REFERENCE}:ro",
        "-v",
        f"{ROOT}:{ROOT}:ro",
        "-v",
        f"{output}:{output}",
        "-v",
        f"{ipc}:{ipc}",
        "-w",
        str(ROOT),
    ]
    for value in (
        "ROS_LOCALHOST_ONLY=1",
        "ROS_DOMAIN_ID=0",
        "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        "CYCLONEDDS_URI=file:///workspace/franka_upper_body_teleop/config/cyclonedds.xml",
        "PYTHONDONTWRITEBYTECODE=1",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        f"ROS_LOG_DIR={output}/ros",
        f"PYTHONPATH={ROOT}" + (":/sdk" if sdk else ""),
    ):
        args += ["-e", value]
    if controller:
        args += [
            "--privileged",
            "--ulimit",
            "rtprio=99",
            "--ulimit",
            "rttime=-1",
            "--ulimit",
            "memlock=-1",
            "-v",
            "/dev:/dev",
            "-v",
            f"{OVERLAY}:/overlay:ro",
        ]
    else:
        args += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    if sdk:
        for package in ("wuji_sdk", "wuji_sdk.libs"):
            if not (SDK / package).is_dir():
                raise ValueError("Missing installed SDK dependency: " + str(SDK / package))
            args += ["-v", f"{SDK / package}:/sdk/{package}:ro"]
    return [*args, "franka-upper-body-teleop:latest", *command]


def launch_commands(args, output, ipc):
    prefix = "pi05-single-" + uuid.uuid4().hex[:8]
    commands = {}
    if args.execute:
        commands["arms"] = docker_command(
            prefix + "-arms",
            output,
            ipc,
            [
                "bash",
                "-c",
                'source "$1"; exec ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:="$2"',
                "pi05",
                "/overlay/install/setup.bash",
                str(output / "workcell.yaml"),
            ],
            controller=True,
        )
        for role in ("gateway", "splitter"):
            commands[role] = docker_command(
                prefix + "-" + role,
                output,
                ipc,
                ["python3", "-m", "experiments.weight_motion_eval.oneshot.ros_boundary", role],
            )
    bridge = [
        "python3",
        "-m",
        "experiments.weight_motion_eval.oneshot.bridge",
        "--runtime",
        str(output / "runtime.json"),
        "--socket",
        str(ipc / "devices.sock"),
    ]
    if args.execute:
        bridge += ["--execute"]
        bridge += (
            ["--supervised-trial"] if args.supervised_trial else ["--qualification", str(args.qualification.resolve())]
        )
    commands["devices"] = docker_command(prefix + "-devices", output, ipc, bridge, sdk=True)
    if args.execute and not args.supervised_trial:
        index = commands["devices"].index("franka-upper-body-teleop:latest")
        directory = args.qualification.resolve().parent
        commands["devices"][index:index] = ["-v", f"{directory}:{directory}:ro"]
    return commands


def preflight_owners(*, execute):
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    if any(
        marker in processes
        for marker in (
            "apps.operator_gui",
            "apps.wuji_ui",
            "adapters.wuji.hand_only",
            "hand_read.py",
            "observation_sources.py hands",
        )
    ):
        raise RuntimeError("An existing hand SDK owner must be closed before this deployment acquires devices")
    containers = subprocess.check_output(
        ["docker", "ps", "--no-trunc", "--format", "{{.Names}} {{.Command}}"], text=True
    )
    if any(marker in containers for marker in ("oneshot.bridge", "hand-control", "adapters.wuji")):
        raise RuntimeError("An existing hand device container must exit before acquiring its SDK devices")
    if execute:
        if any(
            marker in containers
            for marker in (
                "franka-control",
                "robot_control.launch",
                "franka_fr3_arm_controllers.launch",
                "moveit-real",
                "oneshot.bridge",
            )
        ):
            raise RuntimeError("Existing device/control owner detected; refusing a second FCI or SDK owner")
        if "read_arms" in processes:
            raise RuntimeError("An existing direct FCI reader must stop before controller acquisition")


class Children:
    def __init__(self, output):
        self.output, self.children, self.streams = output, [], []

    def launch(self, label, command):
        stream = (self.output / (label + ".log")).open("x")
        self.streams.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        self.children.append((command[command.index("--name") + 1], child))

    def check(self):
        if any(child.poll() is not None for _, child in self.children):
            raise RuntimeError("A deployment process exited; inspect per-process logs")

    def close(self):
        for name, child in reversed(self.children):
            subprocess.run(
                ["docker", "stop", "--signal", "SIGINT", "--timeout", "30", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)
        for stream in self.streams:
            stream.close()


def main(argv=None):
    args = parse_args(argv)
    from experiments.weight_motion_eval.cli import safe_output
    from experiments.weight_motion_eval.reference import deployment_module

    output = safe_output(args.output)
    output.mkdir(parents=True)
    runtime, limits, names = write_runtime(args, output)
    if not args.read_only and not args.execute:
        preview = copy.copy(args)
        preview.execute = True
        preview.qualification = args.qualification or output / "qualification.required.json"
        commands = launch_commands(preview, output, Path("/tmp/pi05-one-RUNTIME"))
        (output / "commands-preview.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "mode": "check",
                    "hardware_output": False,
                    "runtime": str(output / "runtime.json"),
                    "arms": runtime["arms"],
                    "hands": runtime["hands"],
                    "controller_verified": True,
                    "execution_requires": "--execute plus --qualification or --supervised-trial; no devices started",
                },
                indent=2,
            )
        )
        return
    qualification = None
    if args.execute:
        qualification = (
            SupervisedTrial()
            if args.supervised_trial
            else Qualification(args.qualification, controller_sha256=runtime["controller_sha256"])
        )
        (output / "execution-admission.json").write_text(
            json.dumps(
                {
                    "execution_mode": "supervised_trial" if args.supervised_trial else "commissioned",
                    "historical_stop_evidence_waived": args.supervised_trial,
                    "hand_velocity_rad_s": HAND_SPEED_RAD_S,
                    "single_prediction_steps": 50,
                    "physical_stop_qualified": not args.supervised_trial,
                },
                indent=2,
            )
            + "\n"
        )
    preflight_owners(execute=args.execute)
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / ".deployment/oneshot-owner.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ws, model = None, None
        if args.execute:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect

            ws = stack.enter_context(
                connect(args.uri, compression=None, max_size=None, open_timeout=5, close_timeout=2, proxy=None)
            )
            model = msgpack_numpy.unpackb(ws.recv(timeout=3))
            if (
                any(model.get(key) != value for key, value in runtime["checkpoint"].items())
                or model.get("warmed_up") is not True
            ):
                raise ValueError("Wrong checkpoint/normalization or model server not warmed up")
        ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pi05-one-")))
        children = Children(output)
        stack.callback(children.close)
        commands = launch_commands(args, output, ipc)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        for label, command in commands.items():
            children.launch(label, command)
        deadline = time.monotonic() + 45
        while not (ipc / "devices.sock").exists():
            children.check()
            if time.monotonic() > deadline:
                raise RuntimeError("Device bridge startup timed out")
            time.sleep(0.1)
        devices = RemoteDevices(ipc / "devices.sock", execute=args.execute, finish_policy=args.finish_policy)
        stack.callback(devices.close)
        try:
            inventory = devices.call("inventory", timeout=3)
            (output / "devices-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            feedback = None
            while time.monotonic() < deadline:
                children.check()
                try:
                    feedback = devices.feedback(time.monotonic())
                    break
                except RuntimeError:
                    time.sleep(0.05)
            if feedback is None:
                readiness = devices.call("readiness")
                (output / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
                raise RuntimeError(
                    "Missing live arm/hand feedback; read-only mode requires existing arm state publishers"
                )
            if args.read_only:
                (output / "feedback.json").write_text(
                    json.dumps(
                        {
                            "hardware_output": False,
                            "positions": feedback.positions.tolist(),
                            "velocities": feedback.velocities.tolist(),
                        },
                        indent=2,
                    )
                )
                print("Four-device measured feedback received; no motor enable or command sent.")
                return
            from experiments.weight_motion_eval.planner import read_config

            from .live import LiveConsumer

            recorder_module = deployment_module("executor.async_recorder")
            recorder = recorder_module.AsyncRecorder(output)
            try:
                config = read_config(Path(__file__).parents[1] / "config.yaml")
                config["hand_raw_initial_delta_rad"] = qualification.data["hand_raw_initial_delta_rad"]
                consumer = LiveConsumer(devices, ws, model, config, limits, names, output, recorder)
                observe = deployment_module("observe")
                options = [
                    "--arm-source",
                    "ros",
                    "--hand-source",
                    "ros",
                    "--duration",
                    "3",
                    "--poll-interval",
                    "0.005",
                    "--output",
                    str(output / "observation"),
                ]
                if args.start_cameras:
                    options += ["--start-cameras", "--camera-profile", "rgb"]
                observe.main(options, consumer=consumer, single_shot=True)
                if consumer.completed != 1:
                    raise RuntimeError("No fresh inference was admitted")
                print(
                    "One 50-step trajectory completed. "
                    + (
                        "Hands holding; Ctrl-C stops and releases after feedback confirmation."
                        if args.finish_policy == "hold"
                        else "Hands released after settling."
                    ),
                    flush=True,
                )
                while args.finish_policy == "hold":
                    children.check()
                    if devices.hold_error:
                        raise RuntimeError("Hold monitoring failed: " + devices.hold_error)
                    time.sleep(0.1)
            finally:
                # Stop motion/hold before potentially blocking on recorder drain.
                try:
                    devices.stop()
                finally:
                    recorder.close()
        except KeyboardInterrupt:
            print("Stop requested; waiting for device process cleanup.", flush=True)
        finally:
            if args.execute:
                try:
                    result = devices.stop()
                    (output / "stop-request.json").write_text(json.dumps(result, indent=2) + "\n")
                except Exception as error:
                    (output / "stop-request.json").write_text(
                        json.dumps({"physical_stop_confirmed": False, "error": str(error)}) + "\n"
                    )


if __name__ == "__main__":
    main()
