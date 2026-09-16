# ruff: noqa: RUF001 -- Chinese operator messages use native punctuation.
"""Deploy GR00T through the existing FR3/Wuji device boundary.

The default mode is an offline launch preview. Real hardware requires both an
explicit model-manifest SHA256 and the existing supervised/qualified admission.
"""

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

import yaml

from experiments.weight_motion_eval.cli import safe_output
from experiments.weight_motion_eval.oneshot import deploy as hardware
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices
from experiments.weight_motion_eval.oneshot.qualification import Qualification
from experiments.weight_motion_eval.oneshot.qualification import SupervisedTrial
from experiments.weight_motion_eval.reference import deployment_module
from experiments.weight_motion_eval.reference import reference_limits

from .client import GrootClient
from .consumer import GrootConsumer
from .contract import ACTION_ORDER

ROOT = hardware.ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Write an offline launch preview; no sockets or devices")
    modes.add_argument("--server-check", action="store_true", help="Validate GR00T metadata; no device access")
    modes.add_argument("--read-only", action="store_true", help="Read existing device feedback; do not infer or move")
    modes.add_argument("--execute", action="store_true", help="Execute live GR00T predictions on real hardware")
    parser.add_argument("--server", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--expected-model-sha256",
        help="Exact model_manifest_sha256 printed by model_server; mandatory for --execute",
    )
    parser.add_argument("--left-arm-ip", default="172.16.0.2")
    parser.add_argument("--right-arm-ip", default="172.16.1.2")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:7447")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--finish-policy", choices=("hold", "disable"), default="disable")
    parser.add_argument("--rounds", type=int, default=1, help="Number of admitted sixteen-step GR00T chunks")
    parser.add_argument("--guidance-steps", type=int, default=3)
    parser.add_argument("--supervised-trial", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs/weight_motion_eval" / ("groot-" + uuid.uuid4().hex[:10]),
    )
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.rounds <= 1000:
            raise ValueError("rounds must be between 1 and 1000")
        if not 2 <= args.guidance_steps < 16:
            raise ValueError("guidance-steps must be between 2 and 15")
        for address in (args.left_arm_ip, args.right_arm_ip):
            ipaddress.IPv4Address(address)
        for address in (args.wuji_left_address, args.wuji_right_address):
            ip, port = address.rsplit(":", 1)
            ipaddress.IPv4Address(ip)
            if not 1 <= int(port) <= 65535:
                raise ValueError("Hand port must be between 1 and 65535")
        if args.left_arm_ip == args.right_arm_ip or args.wuji_left_address == args.wuji_right_address:
            raise ValueError("Left and right device addresses must be distinct")
        if args.execute and args.expected_model_sha256 is None:
            raise ValueError("--execute requires --expected-model-sha256")
        if args.supervised_trial and (not args.execute or args.qualification is not None):
            raise ValueError("--supervised-trial requires --execute and no --qualification")
        if args.execute and not args.supervised_trial and args.qualification is None:
            raise ValueError("Use --supervised-trial or --qualification for real hardware")
    except (ValueError, AttributeError) as error:
        parser.error(str(error))
    args.hand_control = "slider"
    args.continuous = False
    return args


def stream_options(args):
    return {"rounds": args.rounds, "guidance_steps": args.guidance_steps}


def write_runtime(args, output, model):
    limits, names = reference_limits()
    overlay = hardware.verify_overlay()
    workcell = yaml.safe_load((hardware.REFERENCE / "config/workcell/current.yaml").read_text())
    workcell["LEFT"]["robot_ip"], workcell["RIGHT"]["robot_ip"] = args.left_arm_ip, args.right_arm_ip
    for side in ("LEFT", "RIGHT"):
        if workcell[side]["use_fake_hardware"] != "false":
            raise ValueError("Expected the reference real workcell profile")
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    runtime = {
        "schema_version": 1,
        "reference": str(hardware.REFERENCE),
        "hands": {"left": args.wuji_left_address, "right": args.wuji_right_address},
        "arms": {"left": args.left_arm_ip, "right": args.right_arm_ip},
        "limits": {key: limits[key].tolist() for key in ("lower", "upper")},
        "reference_hashes": limits["sources"],
        "joint_names": names,
        "controller_library": overlay["built_library"],
        "controller_sha256": overlay["built_library_sha256"],
        "checkpoint": {
            key: model.get(key)
            for key in ("backend", "model_manifest_sha256", "statistics_sha256")
            if model.get(key) is not None
        },
        "finish_policy": args.finish_policy,
        "hand_control": args.hand_control,
        "continuous": args.continuous,
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime, limits


def main(argv=None):
    args = parse_args(argv)
    client = None
    if args.execute or args.server_check:
        client = GrootClient(
            args.server,
            expected_model_sha256=args.expected_model_sha256,
            timeout=5.0,
        )
        if args.server_check:
            print(json.dumps(client.metadata, indent=2))
            return
    model = client.metadata if client is not None else {"backend": "groot_n1d7", "server": args.server}
    output = safe_output(args.output)
    output.mkdir(parents=True)
    runtime, limits = write_runtime(args, output, model)
    settings = {
        "backend": "groot_n1d7",
        "source_hz": 30,
        "control_hz": 100,
        "action_horizon": 16,
        "action_order": ACTION_ORDER,
        "units": "radian",
        "action_semantics": "absolute_joint_targets",
        "model_server": args.server,
        "model": model,
        "options": stream_options(args),
        "broker_mode": "serial_only",
        "hardware_output": args.execute,
        "physical_stop_qualified": False,
    }
    (output / "groot-config.json").write_text(json.dumps(settings, indent=2) + "\n")
    if not args.execute and not args.read_only:
        preview = copy.copy(args)
        preview.execute, preview.supervised_trial, preview.qualification = True, True, None
        commands = hardware.launch_commands(preview, output, Path("/tmp/pi05-groot-RUNTIME"))
        (output / "commands-preview.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(json.dumps({"mode": "check", "output": str(output), **settings}, indent=2))
        return
    if args.execute:
        qualification = (
            SupervisedTrial(hand_control="slider")
            if args.supervised_trial
            else Qualification(
                args.qualification,
                controller_sha256=runtime["controller_sha256"],
                hand_control="slider",
            )
        )
        (output / "execution-admission.json").write_text(json.dumps(qualification.data, indent=2) + "\n")
    hardware.preflight_owners(execute=args.execute)
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / ".deployment/oneshot-owner.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pi05-groot-")))
        children = hardware.Children(output)
        stack.callback(children.close)
        commands = hardware.launch_commands(args, output, ipc)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print("【阶段】启动现有 FR3 网关与 Wuji 设备接口", flush=True)
        for label, command in commands.items():
            children.launch(label, command)
        deadline = time.monotonic() + 45
        while not (ipc / "devices.sock").exists():
            children.check()
            if time.monotonic() >= deadline:
                raise RuntimeError("Device bridge startup timed out")
            time.sleep(0.1)
        devices = RemoteDevices(ipc / "devices.sock", execute=args.execute, finish_policy=args.finish_policy)
        stack.callback(devices.close)
        consumer = None
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
                raise RuntimeError("No fresh dual-arm/dual-hand feedback")
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
                    + "\n"
                )
                print("已读取四设备反馈，没有请求模型或下发动作。")
                return
            consumer = GrootConsumer(
                devices,
                client,
                limits,
                output,
                stream_options(args),
                ROOT / "experiments/weight_motion_eval/config.yaml",
            )
            options = [
                "--arm-source",
                "ros",
                "--hand-source",
                "ros",
                "--duration",
                str(args.rounds * 4 + 120),
                "--poll-interval",
                "0.005",
                "--output",
                str(output / "observation"),
            ]
            if args.start_cameras:
                options += ["--start-cameras", "--camera-profile", "rgb"]
            deployment_module("observe").main(options, consumer=consumer, single_shot=True)
            if consumer.completed != 1:
                raise RuntimeError("GR00T serial stream did not complete")
            print(
                "【阶段】GR00T 动作流完成"
                + ("，保持末姿态，Ctrl-C 停止" if args.finish_policy == "hold" else "，双手已释放"),
                flush=True,
            )
            while args.finish_policy == "hold":
                children.check()
                consumer.poll()
                time.sleep(0.05)
        except BaseException as error:
            if consumer is not None and not consumer.completed:
                consumer.report.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if consumer is not None:
                consumer.close()
            if args.execute and not devices.closed:
                devices.stop()


if __name__ == "__main__":
    main()
