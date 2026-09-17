"""Slow execution engine, shared by 54D and 64-to-54 model adapters."""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import ipaddress
import json
from pathlib import Path
import tempfile
import time
import uuid
from fr3_wuji.paths import ROOT
from deployment import hardware
from deployment.profiles import get_profile
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices
from experiments.weight_motion_eval.oneshot.limits import (
    HAND_SPEED_RAD_S, HAND_CURRENT_A, HAND_CONTACT_SECONDS, SLIDER_CURRENT_A, SLIDER_SPEED_RAD_S,
)
from experiments.weight_motion_eval.oneshot.qualification import Qualification, SupervisedTrial


def parse_args(argv=None, *, profile=None):
    profile = profile or get_profile("slow54")
    if profile.engine != "slow":
        raise ValueError("Slow execution requires a slow profile")
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
    parser.add_argument("--checkpoint", type=Path, default=profile.checkpoint)
    parser.add_argument("--uri", default=profile.uri)
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--finish-policy", choices=("hold", "disable"), default="hold")
    parser.add_argument("--hand-control", choices=("slider", "strict"), default="slider")
    parser.add_argument("--continuous", action="store_true", help="Execute fresh prediction prefixes in one held session")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=min(20, profile.action_horizon))
    parser.add_argument("--minimum-time-scale", type=float, default=2.5)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--supervised-trial",
        action="store_true",
        help=f"One operator-supervised {profile.action_horizon}-step run without historical commissioning evidence",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/weight_motion_eval" / ("deployment-" + uuid.uuid4().hex[:10])
    )
    args = parser.parse_args(argv)
    if args.rounds < 1 or not 2 <= args.replan_steps <= profile.action_horizon:
        parser.error(f"rounds must be positive; replan-steps must be between 2 and {profile.action_horizon}")
    if not 1 <= args.minimum_time_scale <= 100:
        parser.error("minimum-time-scale must be between 1 and 100")
    if args.continuous and args.execute and not args.supervised_trial:
        parser.error("Continuous execution currently requires --supervised-trial")
    if args.supervised_trial and (not args.execute or args.qualification is not None):
        parser.error("--supervised-trial requires --execute and cannot be combined with --qualification")
    if profile.camera_profile == "dp3" and args.execute and not args.start_cameras:
        parser.error("DP3 execution requires --start-cameras so the validated depth/point-cloud profile is used")
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


def main(argv=None, *, profile=None):
    profile = profile or get_profile("slow54")
    args = parse_args(argv, profile=profile)
    from experiments.weight_motion_eval.cli import safe_output
    from experiments.weight_motion_eval.reference import deployment_module

    output = safe_output(args.output)
    output.mkdir(parents=True)
    runtime, limits, names = hardware.write_runtime(args, output, profile=profile)
    if not args.read_only and not args.execute:
        preview = copy.copy(args)
        preview.execute = True
        preview.qualification = args.qualification or output / "qualification.required.json"
        commands = hardware.launch_commands(preview, output, Path("/tmp/pi05-one-RUNTIME"))
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
            SupervisedTrial(hand_control=args.hand_control)
            if args.supervised_trial
            else Qualification(
                args.qualification, controller_sha256=runtime["controller_sha256"], hand_control=args.hand_control
            )
        )
        (output / "execution-admission.json").write_text(
            json.dumps(
                {
                    "execution_mode": "supervised_trial" if args.supervised_trial else "commissioned",
                    "historical_stop_evidence_waived": args.supervised_trial,
                    "hand_control": args.hand_control,
                    "hand_velocity_rad_s": SLIDER_SPEED_RAD_S if args.hand_control == "slider" else HAND_SPEED_RAD_S,
                    "hand_measured_speed_stop": args.hand_control == "strict",
                    "hand_current_limit_a": SLIDER_CURRENT_A if args.hand_control == "slider" else HAND_CURRENT_A,
                    "hand_contact_duration_s": HAND_CONTACT_SECONDS if args.hand_control == "slider" else None,
                    "single_prediction_steps": profile.action_horizon,
                    "continuous": args.continuous,
                    "rounds": args.rounds if args.continuous else 1,
                    "execution_steps": args.replan_steps if args.continuous else profile.action_horizon,
                    "physical_stop_qualified": not args.supervised_trial,
                },
                indent=2,
            )
            + "\n"
        )
    hardware.preflight_owners(execute=args.execute)
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
        children = hardware.Children(output)
        stack.callback(children.close)
        commands = hardware.launch_commands(args, output, ipc)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print("【阶段】启动机械臂、手部设备及网关通信", flush=True)
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

            from experiments.weight_motion_eval.oneshot.live import LiveConsumer

            recorder_module = deployment_module("executor.async_recorder")
            recorder = recorder_module.AsyncRecorder(output)
            try:
                config = read_config(
                    ROOT / "experiments/weight_motion_eval/config.yaml",
                    profile=profile.planning_profile,
                    source_hz=profile.action_sample_hz,
                )
                config["hand_control"] = args.hand_control
                config["minimum_time_scale"] = args.minimum_time_scale
                config["hand_raw_initial_delta_rad"] = qualification.data["hand_raw_initial_delta_rad"]
                config["prediction_steps"] = profile.action_horizon
                config["execution_steps"] = args.replan_steps if args.continuous else profile.action_horizon
                if args.continuous:
                    from experiments.weight_motion_eval.oneshot.continuous import ContinuousConsumer
                    consumer = ContinuousConsumer(devices, ws, model, config, limits, names, output, recorder,
                                                  rounds=args.rounds, replan_steps=args.replan_steps)
                    stack.callback(consumer.close)
                else:
                    consumer = LiveConsumer(devices, ws, model, config, limits, names, output, recorder)
                observe = deployment_module("observe")
                options = [
                    "--arm-source",
                    "ros",
                    "--hand-source",
                    "ros",
                    "--duration",
                    str(args.rounds * 90 + 60) if args.continuous else "3",
                    "--poll-interval",
                    "0.005",
                    "--output",
                    str(output / "observation"),
                ]
                if args.start_cameras:
                    options += ["--start-cameras", "--camera-profile", profile.camera_profile]
                print("【阶段】启动实时观测" + ("，等待相机稳定 35 秒" if args.start_cameras else ""), flush=True)
                observe.main(options, consumer=consumer, single_shot=True)
                if consumer.completed != 1:
                    raise RuntimeError("No fresh inference was admitted")
                print(
                    (
                        f"{args.rounds} prediction rounds completed. "
                        if args.continuous
                        else f"One {profile.action_horizon}-step trajectory completed. "
                    )
                    + (
                        "Hands holding; Ctrl-C stops and releases after feedback confirmation."
                        if args.finish_policy == "hold"
                        else "Hands released after settling."
                    ),
                    flush=True,
                )
                while args.finish_policy == "hold":
                    children.check()
                    if args.continuous:
                        consumer.poll()
                    if devices.hold_error:
                        raise RuntimeError("Hold monitoring failed: " + devices.hold_error)
                    time.sleep(0.1)
            finally:
                # Stop motion/hold before potentially blocking on recorder drain.
                try:
                    if args.continuous:
                        consumer.close()
                    devices.stop()
                finally:
                    recorder.close()
        except KeyboardInterrupt:
            print("【阶段】收到停止请求，等待设备停止及清理", flush=True)
        finally:
            if args.execute:
                try:
                    result = devices.stop()
                    (output / "stop-request.json").write_text(json.dumps(result, indent=2) + "\n")
                except Exception as error:
                    (output / "stop-request.json").write_text(
                        json.dumps({"physical_stop_confirmed": False, "error": str(error)}) + "\n"
                    )
