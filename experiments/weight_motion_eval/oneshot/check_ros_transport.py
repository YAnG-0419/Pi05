"""Network-isolated ROS integration check with fake feedback; no robot drivers.

Run inside Docker with --network none and ROS_DOMAIN_ID=197. The namespace
check below prevents accidental use on the host or a host-network container.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    if not Path("/.dockerenv").exists() or set(os.listdir("/sys/class/net")) != {"lo"}:
        raise RuntimeError("This test requires a Docker --network none container")
    if os.environ.get("ROS_DOMAIN_ID") != "197":
        raise RuntimeError("Use the isolated test ROS domain 197")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("accept", "missing_torque", "expired", "partial"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/workspace/franka_upper_body_teleop/ros_ws/src/teleop_core")
    import rclpy
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from teleop_core import contract
    from teleop_interfaces.msg import ArmCommand
    from teleop_interfaces.msg import ArmCommandStatus

    processes = []
    rclpy.init()
    node = rclpy.create_node("pi05_fake_devices")
    seen = {"status": [], "left": [], "right": [], "bus": []}
    for side in ("left", "right"):
        node.create_subscription(
            JointState,
            contract.CONTROLLER_COMMAND_TOPIC.format(side=side),
            lambda msg, side=side: seen[side].append(msg),
            10,
        )
    node.create_subscription(JointState, contract.ARM_COMMAND_TOPIC, lambda msg: seen["bus"].append(msg), 10)
    node.create_subscription(
        ArmCommandStatus, contract.COMMAND_STATUS_TOPIC, lambda msg: seen["status"].append(msg), 10
    )
    source = node.create_publisher(ArmCommand, contract.SOURCE_COMMAND_TOPIC, 1)
    reset = node.create_publisher(
        Bool, contract.RESET_ACTIVE_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    )
    state_pubs = {
        side: node.create_publisher(JointState, contract.ARM_STATE_TOPIC.format(side=side), 1)
        for side in ("left", "right")
    }
    torque_pubs = {
        side: node.create_publisher(JointState, contract.EXTERNAL_TORQUES_TOPIC.format(side=side), 1)
        for side in ("left", "right")
    }
    pose = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]

    def spin(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            for side in ("left", "right"):
                state = JointState()
                state.header.stamp = node.get_clock().now().to_msg()
                state.name = [f"{side}_fr3_joint{i}" for i in range(1, 8)]
                state.position = pose
                state_pubs[side].publish(state)
                if args.case != "missing_torque" or side == "left":
                    torque = JointState()
                    torque.header = state.header
                    torque.name, torque.effort = state.name, [0.0] * 7
                    torque_pubs[side].publish(torque)
            reset.publish(Bool(data=False))
            rclpy.spin_once(node, timeout_sec=0.003)

    def command(sequence, *, expired=False, partial=False):
        msg = ArmCommand()
        created = time.time_ns() - (30_000_000 if expired else 0)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(created, 1_000_000_000)
        msg.header.frame_id = f"pi05v1/{'a' * 32}/{sequence}/{created + 20_000_000}"
        msg.source, msg.session_id, msg.sequence = "openpi", "a" * 32, sequence
        msg.active_sides, msg.joint_names = ["left", "right"], list(contract.COMMAND_JOINT_NAMES)
        msg.positions = pose * 2
        if partial:
            msg.positions[7] = 0.1
        source.publish(msg)
        return msg

    try:
        for role in ("gateway", "splitter"):
            processes.append(  # noqa: PERF401 - retain each child immediately for exception cleanup
                subprocess.Popen([sys.executable, "-m", "experiments.weight_motion_eval.oneshot.ros_boundary", role])
            )
        spin(3)
        assert all(p.poll() is None for p in processes), "ROS boundary failed to start"
        assert node.count_subscribers(contract.SOURCE_COMMAND_TOPIC) == 1
        original = command(0, expired=args.case == "expired", partial=args.case == "partial")
        spin(0.2)
        assert seen["status"], "Missing gateway acknowledgement"
        if args.case == "accept":
            assert list(seen["status"][0].accepted_sides) == ["left", "right"], str(seen["status"][0].faults)
            assert all(len(seen[key]) == 1 for key in ("bus", "left", "right")), seen
            for key in ("bus", "left", "right"):
                assert seen[key][0].header == original.header, "Deadline/header was rewritten"
            # An expired next frame faults and a later fresh frame cannot revive it.
            command(1, expired=True)
            spin(0.1)
            command(2)
            spin(0.1)
            assert all(len(seen[key]) == 1 for key in ("bus", "left", "right"))
            assert all(msg.faults for msg in seen["status"][1:])
        else:
            assert seen["status"][0].faults
            assert not any(seen[key] for key in ("bus", "left", "right"))
        print(
            json.dumps(
                {
                    "case": args.case,
                    "passed": True,
                    "hardware_output": False,
                    "counts": {key: len(value) for key, value in seen.items()},
                }
            )
        )
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
