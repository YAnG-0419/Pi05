"""Host planner -> Unix IPC -> actual ROS gateway/splitter -> fake devices.

No SDK connection or FCI driver is imported. The container role refuses a
network namespace with any interface other than loopback.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np


def container_main(ipc):
    if not Path("/.dockerenv").exists() or set(os.listdir("/sys/class/net")) != {"lo"}:
        raise RuntimeError("Fake-device integration must run with Docker --network none")
    sys.path.insert(0, "/workspace/franka_upper_body_teleop/ros_ws/src/teleop_core")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from teleop_core import contract

    from .bridge import serve_connection
    from .devices import DeviceSession
    from .ros_devices import RosArms
    from .transport import validate_header

    data = json.loads((ipc / "fixture.json").read_text())
    start = np.array(data["start"])
    rclpy.init()
    robot = rclpy.create_node("pi05_fake_robot")
    robot_executor = SingleThreadedExecutor()
    robot_executor.add_node(robot)
    done = threading.Event()
    q = {"left": start[:7].copy(), "right": start[27:34].copy()}
    targets = {side: value.copy() for side, value in q.items()}
    expiry = {"left": 0, "right": 0}
    pubs = {side: robot.create_publisher(JointState, contract.ARM_STATE_TOPIC.format(side=side), 1) for side in q}
    torques = {
        side: robot.create_publisher(JointState, contract.EXTERNAL_TORQUES_TOPIC.format(side=side), 1) for side in q
    }
    reset = robot.create_publisher(
        Bool, contract.RESET_ACTIVE_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    )

    def command(side, msg):
        created = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        _, _, expiry[side] = validate_header(msg.header.frame_id, created, time.time_ns())
        targets[side] = np.array(msg.position)

    for side in q:
        robot.create_subscription(
            JointState,
            contract.CONTROLLER_COMMAND_TOPIC.format(side=side),
            lambda msg, side=side: command(side, msg),
            1,
        )
    last = time.monotonic()

    def publish():
        nonlocal last
        now = time.monotonic()
        dt, last = now - last, now
        for side, current in q.items():
            if expiry[side] and time.time_ns() >= expiry[side]:
                targets[side] = current.copy()
            before = current.copy()
            current[:] += (targets[side] - current) * (1 - np.exp(-dt / 0.03))
            msg = JointState()
            msg.header.stamp = robot.get_clock().now().to_msg()
            msg.name = [f"{side}_fr3_joint{i}" for i in range(1, 8)]
            msg.position, msg.velocity = current.tolist(), ((current - before) / dt).tolist()
            pubs[side].publish(msg)
            msg.effort = [0.0] * 7
            torques[side].publish(msg)
        reset.publish(Bool(data=False))

    robot.create_timer(0.002, publish)

    def robot_spin():
        while not done.is_set():
            robot_executor.spin_once(timeout_sec=0.005)

    thread = threading.Thread(target=robot_spin)
    thread.start()
    processes = []
    arms, session = None, None

    class FakeHand:
        def __init__(self, position):
            self.q, self.target, self.dq = position.copy(), position.copy(), np.zeros(20)
            self.updated, self.latest_received = time.monotonic(), time.monotonic()
            self.owns_enable, self.stop_count, self.last = False, 0, None

        def identity(self):
            return {"fake": True}

        def poll(self, now, wall):
            dt = now - self.updated
            if dt > 0:
                before = self.q.copy()
                self.q += (self.target - self.q) * (1 - np.exp(-dt / 0.03))
                self.dq = (self.q - before) / dt
                self.updated = now
            self.latest_received = now
            return self.q.copy(), self.dq.copy(), now

        def enable(self, *, qualification, cancelled=lambda: False):
            self.owns_enable = True

        def submit(self, position, **kwargs):
            if time.monotonic() >= kwargs["valid_until"]:
                raise ValueError("Fake hand received an expired frame")
            self.target = np.array(position)

        def emergency_stop(self):
            self.stop_count += 1
            self.target = self.q.copy()

        def disable_after_stop(self, now):
            self.owns_enable = False

        def close_readonly(self):
            pass

    try:
        for role in ("gateway", "splitter"):
            processes.append(  # noqa: PERF401 - preserve each child for partial-start cleanup
                subprocess.Popen([sys.executable, "-m", "experiments.weight_motion_eval.oneshot.ros_boundary", role])
            )
        arms = RosArms()
        hands = {"left": FakeHand(start[7:27]), "right": FakeHand(start[34:54])}
        # This fake qualification object never reaches HandOwner/real SDK.
        session = DeviceSession(
            arms, hands, data["limits"], execute=True, qualification=SimpleNamespace(check_hand=lambda *args: None)
        )
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            arms.spin()
            arms.publish_hands(hands)
            time.sleep(0.001)
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
            server.bind(str(ipc / "devices.sock"))
            os.chmod(ipc / "devices.sock", 0o666)
            server.listen(1)
            conn, _ = server.accept()
            with conn:
                serve_connection(
                    conn, session, stopping=done.is_set, pump=lambda: (arms.spin(), arms.publish_hands(hands))
                )
    finally:
        if session:
            session.close()
            (ipc / "device-result.json").write_text(
                json.dumps(
                    {
                        "state": session.state,
                        "fault": session.fault,
                        "hardware_output": False,
                        "stop_confirmed_on_fake_devices": session.stop_confirmed,
                        "hand_stops": {side: hand.stop_count for side, hand in session.hands.items()},
                    }
                )
            )
        if arms:
            arms.close()
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=5)
        done.set()
        thread.join(2)
        robot_executor.shutdown()
        robot.destroy_node()
        rclpy.shutdown()


def host_main(output, finish_policy):
    from experiments.weight_motion_eval.cli import safe_output
    from experiments.weight_motion_eval.planner import build_plan
    from experiments.weight_motion_eval.planner import load_record
    from experiments.weight_motion_eval.planner import read_config
    from experiments.weight_motion_eval.reference import reference_limits

    from .core import Admission
    from .core import OneShot
    from .deploy import REFERENCE
    from .deploy import ROOT
    from .ipc import RemoteDevices
    from .runner import run_live

    output = safe_output(output)
    output.mkdir(parents=True)
    limits, names = reference_limits()
    raw, start, source = load_record(
        ROOT / "logs/weight_motion_eval/new19999-restored-20260910/live/inference-0000.npz"
    )
    plan = build_plan(raw, start, read_config(Path(__file__).parents[1] / "config.yaml"), limits, names)
    with tempfile.TemporaryDirectory(prefix="pi05-stack-") as directory:
        ipc = Path(directory)
        (ipc / "fixture.json").write_text(
            json.dumps({"start": start.tolist(), "limits": {key: limits[key].tolist() for key in ("lower", "upper")}})
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cpuset-cpus",
            "0-7,16-23",
            "-v",
            f"{ROOT}:{ROOT}:ro",
            "-v",
            f"{REFERENCE}:/workspace/franka_upper_body_teleop:ro",
            "-v",
            f"{ipc}:{ipc}",
            "-w",
            str(ROOT),
            "-e",
            "ROS_DOMAIN_ID=197",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"PYTHONPATH={ROOT}",
            "franka-upper-body-teleop:latest",
            "python3",
            "-m",
            "experiments.weight_motion_eval.oneshot.check_device_stack",
            "--container",
            str(ipc),
        ]
        with (output / "container.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            devices = None
            try:
                deadline = time.monotonic() + 15
                while not (ipc / "devices.sock").exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Fake stack failed to start; inspect container.log")
                    time.sleep(0.02)
                timings = []

                class TimedDevices(RemoteDevices):
                    def _call(self, operation, **kwargs):
                        before = time.monotonic()
                        try:
                            return super()._call(operation, **kwargs)
                        finally:
                            timings.append((operation, time.monotonic() - before))

                devices = TimedDevices(ipc / "devices.sock", execute=True, finish_policy=finish_policy)
                now = time.monotonic()
                player = OneShot(
                    plan, Admission(now - 0.15, now - 0.10, now - 0.05, "fake-only", "historical-simulation", 0), now
                )
                events = []
                run_live(
                    player,
                    devices,
                    stop_requested=lambda: False,
                    emit=lambda frame, feedback: events.append(frame.created),
                )
                if player.state != "complete":
                    raise RuntimeError("Fake stack did not complete: " + str(player.reason))
                if finish_policy == "hold":
                    time.sleep(0.6)
                    if devices.hold_error:
                        raise RuntimeError("Fake hand hold monitor failed: " + devices.hold_error)
                intervals = np.diff(events)
                report = {
                    "hardware_output": False,
                    "source": source,
                    "state": player.state,
                    "finish_policy": finish_policy,
                    "frames": len(events),
                    "output_hz": float(1 / np.mean(intervals)),
                    "interval_max_seconds": float(intervals.max()),
                    "transitions": player.transitions,
                }
                (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
            finally:
                if devices:
                    (output / "rpc-timings.json").write_text(json.dumps(timings))
                if devices:
                    devices.close()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
                if (ipc / "device-result.json").exists():
                    (output / "device-result.json").write_bytes((ipc / "device-result.json").read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finish-policy", choices=("hold", "disable"), default="disable")
    args = parser.parse_args()
    if args.container:
        container_main(args.container)
    elif args.output:
        host_main(args.output, args.finish_policy)
    else:
        parser.error("Specify --output for the host test")


if __name__ == "__main__":
    main()
