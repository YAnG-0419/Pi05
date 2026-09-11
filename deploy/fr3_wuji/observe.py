"""Capture observations; optionally feed a recording callback. No robot commands."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
from observation import IMAGE_KEYS
from observation import ObservationBuffer
from observation import ObservationUnavailableError
from observation import joint_names
from observation import read_record
import yaml

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("/home/user/lpy/gello-retarget")
IMAGE = "franka-upper-body-teleop:latest"


def prepare_camera_profile(cameras, output, profile):
    """Create per-run camera settings; the reference repository remains read-only."""
    if profile == "recording":
        return "/workspace/franka_upper_body_teleop/data_collection/config/cameras.yaml", {}
    if profile != "rgb":
        raise ValueError(f"Unknown camera profile: {profile}")
    directory = output / "camera-config"
    directory.mkdir()
    deployment = {
        "camera_bringup": {key: dict(value) if isinstance(value, dict) else value for key, value in cameras.items()}
    }
    changes = {}
    for camera in IMAGE_KEYS:
        source = REFERENCE / "ros_ws/src/teleop_camera_bringup/config" / cameras[camera]["config_file"]
        params = yaml.safe_load(source.read_text())
        overrides = {"enable_depth": False, "enable_frame_sync": False, "frame_aggregate_mode": "disable"}
        params.update(overrides)
        (directory / f"{camera}.yaml").write_text(yaml.safe_dump(params))
        deployment["camera_bringup"][camera]["config_file"] = f"/logs/camera-config/{camera}.yaml"
        changes[camera] = {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "overrides": overrides,
        }
    (directory / "cameras.yaml").write_text(yaml.safe_dump(deployment))
    return "/logs/camera-config/cameras.yaml", changes


def preflight(args):
    if args.duration <= 0 or not np.isfinite(args.duration):
        raise ValueError("Duration must be positive and finite")
    contract = yaml.safe_load((REFERENCE / "data_collection/contract/dataset_contract.yaml").read_text())
    for source in ("left_arm", "right_arm", "left_hand", "right_hand"):
        if tuple(contract["contract_joint_names"][source]) != joint_names(source):
            raise ValueError(f"Reference contract changed: {source}")
    if contract["units"] != "radian":
        raise ValueError("Reference units changed")
    cameras = yaml.safe_load((REFERENCE / "data_collection/config/cameras.yaml").read_text())["camera_bringup"]
    if [cameras[key]["semantic"] for key in IMAGE_KEYS] != ["head", "left_wrist", "right_wrist"]:
        raise ValueError("Camera semantics do not match training")
    if len({cameras[key]["serial_number"] for key in IMAGE_KEYS}) != 3:
        raise ValueError("Camera serial numbers must be distinct")
    containers = subprocess.check_output(
        ["docker", "ps", "--format", "{{.Names}} {{.Command}}", "--no-trunc"], text=True
    )
    if args.arm_source == "direct" and any(
        s in containers for s in ("franka-control", "robot_control.launch", "moveit-real")
    ):
        raise ValueError("Existing arm control detected. Use --arm-source ros; never create a second FCI owner.")
    if args.start_cameras:
        if subprocess.run(["pgrep", "-x", "OrbbecViewer"], stdout=subprocess.DEVNULL, check=False).returncode == 0:
            raise ValueError("Close OrbbecViewer before starting cameras")
        if any(s in containers for s in ("data-collection", "orbbec", "camera")):
            raise ValueError("Existing camera container detected; omit --start-cameras")
    if args.hand_source == "sdk":
        processes = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        for line in processes.splitlines():
            if any(s in line for s in ("apps.operator_gui", "apps.wuji_ui", "adapters.wuji.hand_only", "hand_read.py")):
                raise ValueError(
                    "Possible existing Hand SDK owner detected; use --hand-source ros or stop its owner yourself"
                )
    return cameras


def compile_reader():
    build = ROOT / ".deployment/observation-build"
    build.mkdir(parents=True, exist_ok=True)
    source = ROOT / "deploy/fr3_wuji/read_arms.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mark = build / "source.sha256"
    if not (build / "read_arms").exists() or not mark.exists() or mark.read_text() != digest:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--cpuset-cpus",
                "0-7,16-23",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{source}:/read_arms.cpp:ro",
                "-v",
                f"{build}:/build",
                "--entrypoint",
                "g++",
                IMAGE,
                "-std=c++17",
                "-O2",
                "/read_arms.cpp",
                "-lfranka",
                "-pthread",
                "-o",
                "/build/read_arms",
            ],
            check=True,
        )
        mark.write_text(digest)
    return build


def main(argv=None, *, consumer=None, single_shot=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-source", choices=("ros", "direct"), default="ros")
    parser.add_argument("--hand-source", choices=("ros", "sdk"), default="ros")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--camera-profile", choices=("recording", "rgb"), default="recording")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--poll-interval", type=float, default=1 / 30)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/fr3_wuji/observations")
    args = parser.parse_args(argv)
    if not np.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("--poll-interval must be positive and finite")
    if args.camera_profile != "recording" and not args.start_cameras:
        parser.error("--camera-profile only configures cameras started with --start-cameras")
    cameras = preflight(args)
    build = compile_reader() if args.arm_source == "direct" else ROOT / ".deployment"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    camera_config, camera_overrides = prepare_camera_profile(cameras, output, args.camera_profile)
    workcell = yaml.safe_load((REFERENCE / "config/workcell/current.yaml").read_text())
    store = ObservationBuffer()
    errors, last_invalid = [], {}
    connections, processes, log_streams = [], [], []
    stopping = threading.Event()
    name = "pi05-observe-" + uuid.uuid4().hex[:8]
    report = {
        "hardware_output": False,
        "inference_performed": False,
        "inference_requested": consumer is not None,
        "arm_source": args.arm_source,
        "hand_source": args.hand_source,
        "cameras": cameras,
        "camera_profile": args.camera_profile,
        "camera_overrides": camera_overrides,
        "duration_seconds": args.duration,
        "poll_interval_seconds": args.poll_interval,
        "limits": {
            "sample_max_age_seconds": store.max_age,
            "max_skew_seconds": store.max_skew,
            "latest_state_max_age_seconds": store.state_live_age,
        },
        "state_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "units": "radian",
        "alignment": "host clock; direct FR3 stamp approximated by read receipt time",
    }

    def read_connection(conn):
        try:
            with conn.makefile("rb") as stream:
                while not stopping.is_set():
                    record, payload = read_record(stream)
                    try:
                        store.add(record, payload)
                    except (ValueError, KeyError, TypeError) as error:
                        last_invalid[record.get("source", "unknown")] = str(error)
        except Exception as error:
            if not stopping.is_set():
                errors.append(str(error))

    def accept(server):
        while not stopping.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            connections.append(conn)
            threading.Thread(target=read_connection, args=(conn,), daemon=True).start()

    def launch(command, label):
        stream = (output / (label + ".txt")).open("w")
        log_streams.append(stream)
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    saved, rejected, accepted = [], Counter(), 0
    last_target = None
    try:
        with tempfile.TemporaryDirectory(prefix="pi05-obs-") as ipc, socket.socket(socket.AF_UNIX) as server:
            socket_path = str(Path(ipc) / "feed.sock")
            server.bind(socket_path)
            server.listen(4)
            server.settimeout(0.5)
            threading.Thread(target=accept, args=(server,), daemon=True).start()
            command = [
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
                "0-7,16-23",
                "-v",
                f"{REFERENCE}:/workspace/franka_upper_body_teleop:ro",
                "-v",
                f"{ROOT / 'deploy/fr3_wuji'}:/code:ro",
                "-v",
                f"{build}:/build:ro",
                "-v",
                f"{ipc}:/feed",
                "-v",
                f"{output}:/logs",
                "-w",
                "/tmp",
            ]
            if args.start_cameras:
                command += ["--privileged", "-v", "/dev:/dev"]
            for value in (
                "ROS_LOCALHOST_ONLY=1",
                "ROS_DOMAIN_ID=0",
                "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
                "CYCLONEDDS_URI=file:///workspace/franka_upper_body_teleop/config/cyclonedds.xml",
                "ORBBEC_SDK_LICENSE_ACCEPTED=YES",
                "ROS_LOG_DIR=/logs/ros",
                "PYTHONDONTWRITEBYTECODE=1",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
            ):
                command += ["-e", value]
            command += [
                IMAGE,
                "python3",
                "/code/observation_sources.py",
                "ros",
                "--socket",
                "/feed/feed.sock",
                "--arm-source",
                args.arm_source,
                "--hand-source",
                args.hand_source,
                "--camera-config",
                camera_config,
                "--left-arm",
                workcell["LEFT"]["robot_ip"],
                "--right-arm",
                workcell["RIGHT"]["robot_ip"],
            ]
            if args.start_cameras:
                command.append("--start-cameras")
            launch(command, "ros-producer")
            if args.hand_source == "sdk":
                launch(
                    [
                        "/home/user/miniconda3/envs/gello-upper-body-teleop/bin/python",
                        "-B",
                        str(ROOT / "deploy/fr3_wuji/observation_sources.py"),
                        "hands",
                        "--socket",
                        socket_path,
                    ],
                    "hand-producer",
                )
            started, deadline, measurement_start = time.monotonic(), time.monotonic() + 95, None
            next_update = started + 15
            first, last = None, None
            while True:
                now = time.monotonic()
                if errors or any(process.poll() is not None for process in processes):
                    raise RuntimeError(f"Observation producer stopped: {errors}; inspect producer logs")
                if measurement_start is None and now > deadline:
                    raise RuntimeError(f"No synchronized observation within 95s: {dict(rejected)}, {last_invalid}")
                if measurement_start is not None and now - measurement_start >= args.duration:
                    break
                if args.start_cameras and now - started < 35:
                    time.sleep(0.02)
                    continue
                if consumer is not None and hasattr(consumer, "poll"):
                    consumer.poll(store, output)
                try:
                    obs, metadata = store.snapshot()
                    if measurement_start is None:
                        measurement_start = now
                        print(f"First synchronized real observation after {now - started:.1f}s", flush=True)
                    if metadata["target_stamp"] != last_target:
                        accepted += 1
                        last_target = metadata["target_stamp"]
                        saved.append(metadata)
                        if first is None:
                            first = obs
                        last = obs
                        if consumer is not None:
                            consumer(store, obs, metadata, output)
                            report["inference_performed"] = consumer.completed > 0
                            if single_shot and consumer.completed == 1:
                                break
                except ObservationUnavailableError as error:
                    rejected[str(error)] += 1
                if now >= next_update:
                    print(f"Accepted {accepted} distinct observations; rejected polls: {dict(rejected)}", flush=True)
                    next_update = now + 15
                time.sleep(args.poll_interval)
            if first is not None:
                np.savez_compressed(output / "first.npz", **first)
                np.savez_compressed(output / "last.npz", **last)
            enough = accepted >= 10 or (single_shot and consumer is not None and consumer.completed == 1)
            report["status"] = "captured" if enough else "insufficient_observations"
            report["accepted_observations"] = accepted
            report["observations"] = saved
            report["rejected_polls"] = dict(rejected)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["inference_performed"] = consumer is not None and consumer.completed > 0
        stopping.set()
        for process in processes[1:]:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        subprocess.run(
            ["docker", "stop", "--signal", "SIGINT", "--timeout", "20", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for conn in connections:
            conn.close()
        for stream in log_streams:
            stream.close()
        report["source_samples_received"] = store.accepted
        report["invalid_source_samples"] = store.invalid
        report["last_invalid_reason"] = last_invalid
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"Report: {output / 'report.json'}", flush=True)
    if report["status"] != "captured":
        raise RuntimeError(report["status"])


if __name__ == "__main__":
    main()
