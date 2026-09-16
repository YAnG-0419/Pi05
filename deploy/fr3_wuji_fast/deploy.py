# ruff: noqa: RUF001 -- Chinese operator messages use native punctuation.
"""Independent normal-speed 19999 deployment. Default: offline configuration check."""

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

from experiments.weight_motion_eval.cli import safe_output
from experiments.weight_motion_eval.oneshot import deploy as hardware
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices
from experiments.weight_motion_eval.oneshot.limits import ARM_TRACKING_TOLERANCE_RAD
from experiments.weight_motion_eval.oneshot.limits import HAND_KD
from experiments.weight_motion_eval.oneshot.limits import HAND_KP
from experiments.weight_motion_eval.oneshot.limits import SLIDER_CURRENT_A
from experiments.weight_motion_eval.oneshot.limits import SLIDER_SPEED_RAD_S
from experiments.weight_motion_eval.oneshot.qualification import Qualification
from experiments.weight_motion_eval.oneshot.qualification import SupervisedTrial
from experiments.weight_motion_eval.reference import deployment_module

from .consumer import FastConsumer
from .timeline import ARM_SPEED_RAD_S
from .timeline import Timeline

ROOT = hardware.ROOT
GATEWAY_ARM_SPEED_RAD_S = 1.2
DEFAULT_CHECKPOINT = ROOT / "checkpoints/19999_269/19999"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Write launch preview only; no model or device connections")
    modes.add_argument("--read-only", action="store_true", help="Read existing feedback through the original bridge")
    modes.add_argument(
        "--execute", action="store_true", help="Start real controllers and execute live model predictions"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--uri", default="ws://127.0.0.1:8001")
    parser.add_argument("--left-arm-ip", default="172.16.0.2")
    parser.add_argument("--right-arm-ip", default="172.16.1.2")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:7447")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--finish-policy", choices=("hold", "disable"), default="disable")
    parser.add_argument("--broker-mode", choices=("rtg", "serial"), default="rtg")
    parser.add_argument("--rounds", type=int, default=50, help="Finite number of admitted model chunks")
    parser.add_argument("--guidance-steps", type=int, default=3)
    parser.add_argument("--trigger-fraction", type=float, default=0.5)
    parser.add_argument("--supervised-trial", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/weight_motion_eval" / ("fast-" + uuid.uuid4().hex[:10])
    )
    args = parser.parse_args(argv)
    # Validate before creating files or importing any hardware driver.
    from types import SimpleNamespace

    import numpy as np

    try:
        Timeline(
            SimpleNamespace(actions=np.zeros((50, 54)), admission=SimpleNamespace(request_id="check")),
            **stream_options(args),
        )
        for address in (args.left_arm_ip, args.right_arm_ip):
            ipaddress.IPv4Address(address)
        for address in (args.wuji_left_address, args.wuji_right_address):
            ip, port = address.rsplit(":", 1)
            ipaddress.IPv4Address(ip)
            if not 1 <= int(port) <= 65535:
                raise ValueError("Hand port must be 1..65535")
        if args.left_arm_ip == args.right_arm_ip or args.wuji_left_address == args.wuji_right_address:
            raise ValueError("Left/right device addresses must be distinct")
        if args.supervised_trial and (not args.execute or args.qualification is not None):
            raise ValueError("--supervised-trial requires --execute and no --qualification")
        if args.execute and not args.supervised_trial and args.qualification is None:
            raise ValueError("Use the existing --supervised-trial or --qualification execution mode")
    except (ValueError, AttributeError) as error:
        parser.error(str(error))
    # Attributes consumed by original launch/runtime helpers; no globals patched.
    args.hand_control, args.continuous = "slider", False
    return args


def stream_options(args):
    return {
        "rounds": args.rounds,
        "mode": args.broker_mode,
        "guidance_steps": args.guidance_steps,
        "trigger_fraction": args.trigger_fraction,
    }


def launch_commands(args, output, ipc):
    commands = hardware.launch_commands(args, output, ipc)
    if "gateway" in commands:
        commands["gateway"] += ["--arm-speed-rad-s", str(GATEWAY_ARM_SPEED_RAD_S)]
    return commands


def main(argv=None):
    args = parse_args(argv)
    output = safe_output(args.output)
    output.mkdir(parents=True)
    runtime, limits, _ = hardware.write_runtime(args, output)
    runtime["arm_speed_rad_s"] = ARM_SPEED_RAD_S
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    settings = {
        "source_hz": 30,
        "control_hz": 100,
        "playback_time_scale": 1.0,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "units": "radian",
        "model_uri": args.uri,
        "options": stream_options(args),
        "initial_approach": "existing bounded quintic approach only",
        "device_limits_inherited": {
            "arm_speed_rad_s": ARM_SPEED_RAD_S,
            "gateway_arm_speed_ceiling_rad_s": GATEWAY_ARM_SPEED_RAD_S,
            "arm_tracking_rad": ARM_TRACKING_TOLERANCE_RAD,
            "slider_speed_rad_s": SLIDER_SPEED_RAD_S,
            "hand_current_a": SLIDER_CURRENT_A,
            "kp": HAND_KP,
            "kd": HAND_KD,
            "frame_lifetime_ms": 20,
        },
        "hardware_output": args.execute,
        "physical_stop_qualified": False,
    }
    (output / "fast-config.json").write_text(json.dumps(settings, indent=2) + "\n")
    if not args.execute and not args.read_only:
        preview = copy.copy(args)
        preview.execute, preview.supervised_trial, preview.qualification = True, True, None
        commands = launch_commands(preview, output, Path("/tmp/pi05-fast-RUNTIME"))
        (output / "commands-preview.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "mode": "check",
                    "hardware_output": False,
                    "output": str(output),
                    "arms": runtime["arms"],
                    "hands": runtime["hands"],
                    "checkpoint": runtime["checkpoint"],
                    **settings,
                },
                indent=2,
            )
        )
        return
    if args.execute:
        qualification = (
            SupervisedTrial(hand_control="slider")
            if args.supervised_trial
            else Qualification(
                args.qualification, controller_sha256=runtime["controller_sha256"], hand_control="slider"
            )
        )
        (output / "execution-admission.json").write_text(json.dumps(qualification.data, indent=2) + "\n")
    hardware.preflight_owners(execute=args.execute)
    with ExitStack() as stack:
        # Same ownership lock as slow and 30000: never acquire a second device session.
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
                raise ValueError("Wrong 19999 checkpoint/normalization or model server not warmed up")
        ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pi05-fast-")))
        children = hardware.Children(output)
        stack.callback(children.close)
        commands = launch_commands(args, output, ipc)
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
        devices = RemoteDevices(
            ipc / "devices.sock",
            execute=args.execute,
            finish_policy=args.finish_policy,
            arm_speed_rad_s=ARM_SPEED_RAD_S,
        )
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
                raise RuntimeError(
                    "No fresh dual-arm/dual-hand feedback; read-only requires existing arm state publishers"
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
                    + "\n"
                )
                print("已读取四设备反馈，没有使能或下发动作。")
                return
            consumer = FastConsumer(
                devices,
                ws,
                model,
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
                raise RuntimeError("Normal-speed stream did not complete")
            print(
                "【阶段】正常速度动作流完成"
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
            # Stop control before slow logging/observation cleanup or container teardown.
            if consumer is not None:
                consumer.close()
            if args.execute and not devices.closed:
                devices.stop()


if __name__ == "__main__":
    main()
