"""Isolated ROS deadline-preserving gateway and splitter; no auto-launch.

Use only with the Pi05 controller overlay. The original workcell gateway and
splitter must not be running alongside these nodes.
"""

import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np

from .transport import validate_header


class ArmBoundary:
    def __init__(self, gate):
        self.gate = gate
        self.session = None
        self.sequence = None
        self.fault = None

    def accept(self, message, measured, torques, now_mono, now_wall_ns):
        if self.fault:
            raise ValueError("Arm boundary fault is latched: " + self.fault)
        try:
            created = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            run, sequence, _ = validate_header(message.header.frame_id, created, now_wall_ns)
            if message.source != "openpi" or message.session_id != run or message.sequence != sequence:
                raise ValueError("Source/session/sequence does not match deadline envelope")
            if list(message.active_sides) != ["left", "right"]:
                raise ValueError("Single-shot dual-arm commands require both sides")
            if self.session is None:
                if sequence != 0:
                    raise ValueError("First arm command must have sequence zero")
            elif run != self.session or sequence != self.sequence + 1:
                raise ValueError("Foreign, repeated or skipped arm command")
            if any(torques.get(side) is None for side in ("left", "right")):
                raise ValueError("Both external-torque estimates are required")
            # Validate a copy so partial acceptance never changes the live gate.
            candidate = copy.deepcopy(self.gate)
            result = candidate.validate(
                message.active_sides,
                message.joint_names,
                message.positions,
                measured,
                now_mono,
                external_torques=torques,
            )
            if result is None or candidate.side_faults or len(result.positions) != 14:
                raise ValueError("Dual-arm gate rejected all/part of command: " + str(candidate.side_faults))
            if not np.allclose(result.positions, message.positions, atol=1e-9, rtol=0):
                raise ValueError("Gateway contact/slew protection changed the planned target")
            self.gate, self.session, self.sequence = candidate, run, sequence
            return result
        except ValueError as error:
            self.fault = str(error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("gateway", "splitter"))
    parser.add_argument("--reference", type=Path, default=Path("/workspace/franka_upper_body_teleop"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference / "ros_ws/src/teleop_core"))
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from teleop_core import contract
    from teleop_core.joint_state import ordered_arm_positions
    from teleop_core.joint_state import ordered_external_torques
    from teleop_core.joint_state import ordered_joint_positions
    from teleop_core.safety import CommandSafetyGate
    from teleop_interfaces.msg import ArmCommand
    from teleop_interfaces.msg import ArmCommandStatus
    import yaml

    params = yaml.safe_load((args.reference / "config/modes/teleop_control.yaml").read_text())["teleop_safety_gateway"][
        "ros__parameters"
    ]
    if params["max_joint_speed"] != 0.7 or params["max_initial_delta"] != 0.05:
        raise ValueError("Reference gateway limits changed")

    class Gateway(Node):
        def __init__(self):
            super().__init__("pi05_oneshot_gateway")
            self.boundary = ArmBoundary(
                CommandSafetyGate(
                    **{
                        k: params[k]
                        for k in ("max_joint_speed", "max_initial_delta", "nominal_dt", "contact_torque_thresholds")
                    }
                )
            )
            self.states, self.torques = {}, {}
            self.reset_active = None
            self.offset = time.time() - time.monotonic()
            self.output = self.create_publisher(JointState, contract.ARM_COMMAND_TOPIC, 1)
            self.audit = self.create_publisher(JointState, contract.VALIDATED_COMMAND_TOPIC, 1)
            self.status = self.create_publisher(ArmCommandStatus, contract.COMMAND_STATUS_TOPIC, 10)
            for side in ("left", "right"):
                self.create_subscription(
                    JointState,
                    contract.ARM_STATE_TOPIC.format(side=side),
                    lambda msg, side=side: self.feedback(side, msg, torque=False),
                    qos_profile_sensor_data,
                )
                self.create_subscription(
                    JointState,
                    contract.EXTERNAL_TORQUES_TOPIC.format(side=side),
                    lambda msg, side=side: self.feedback(side, msg, torque=True),
                    qos_profile_sensor_data,
                )
            reset_qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
            self.create_subscription(Bool, contract.RESET_ACTIVE_TOPIC, self.reset, reset_qos)
            self.create_subscription(ArmCommand, contract.SOURCE_COMMAND_TOPIC, self.command, 1)

        def reset(self, msg):
            self.reset_active = bool(msg.data)
            if self.reset_active and self.boundary.session:
                self.boundary.fault = "Reset interrupted one-shot execution"

        def feedback(self, side, msg, torque):
            try:
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                age = time.time() - stamp
                if not 0 <= age <= 0.15:
                    raise ValueError("stale feedback source timestamp")
                values = (
                    ordered_external_torques(msg.name, msg.effort)
                    if torque
                    else ordered_arm_positions(msg.name, msg.position, side)
                )
                if values is None or np.asarray(values).shape != (7,) or not np.isfinite(values).all():
                    raise ValueError("Incomplete or non-finite arm feedback")
                (self.torques if torque else self.states)[side] = (values, time.monotonic() - age)
            except ValueError:
                (self.torques if torque else self.states).pop(side, None)

        def command(self, msg):
            status = ArmCommandStatus()
            status.header = copy.deepcopy(msg.header)
            status.source, status.session_id, status.sequence = msg.source, msg.session_id, msg.sequence
            try:
                now = time.monotonic()
                if abs(time.time() - now - self.offset) > 0.05:
                    raise ValueError("Host clock offset changed")
                if self.reset_active is not False:
                    raise ValueError("Reset state missing or active")
                if self.count_publishers(contract.ARM_COMMAND_TOPIC) != 1:
                    raise ValueError("Competing arm command-bus publisher")
                if self.count_publishers(contract.SOURCE_COMMAND_TOPIC) != 1:
                    raise ValueError("Competing policy/teleoperation command source")
                for group in (self.states, self.torques):
                    if any(side not in group or not 0 <= now - group[side][1] <= 0.15 for side in ("left", "right")):
                        raise ValueError("Arm state/external torque missing or expired")
                validated = self.boundary.accept(
                    msg,
                    np.concatenate([self.states[s][0] for s in ("left", "right")]),
                    {s: self.torques[s][0] for s in ("left", "right")},
                    now,
                    time.time_ns(),
                )
                created = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                validate_header(msg.header.frame_id, created, time.time_ns())
                output = JointState()
                output.header = copy.deepcopy(msg.header)  # Never re-stamp queued commands.
                output.name, output.position = list(validated.names), list(validated.positions)
                self.output.publish(output)
                self.audit.publish(output)
                status.accepted_sides = ["left", "right"]
            except ValueError as error:
                self.boundary.fault = str(error)
                status.faults = [str(error)]
                self.get_logger().error(str(error))
            self.status.publish(status)

    class Splitter(Node):
        def __init__(self):
            super().__init__("pi05_oneshot_splitter")
            self.command_outputs = {
                s: self.create_publisher(JointState, contract.CONTROLLER_COMMAND_TOPIC.format(side=s), 1)
                for s in ("left", "right")
            }
            self.create_subscription(JointState, contract.ARM_COMMAND_TOPIC, self.command, 1)
            self.fault = False

        def command(self, msg):
            if self.fault:
                return
            try:
                created = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                validate_header(msg.header.frame_id, created, time.time_ns())
                outputs = []
                for side, names in (
                    ("left", contract.LEFT_COMMAND_JOINT_NAMES),
                    ("right", contract.RIGHT_COMMAND_JOINT_NAMES),
                ):
                    if self.count_publishers(contract.CONTROLLER_COMMAND_TOPIC.format(side=side)) != 1:
                        raise ValueError("Competing controller-command publisher")
                    q = ordered_joint_positions(msg.name, msg.position, names)
                    if q is None:
                        raise ValueError("Incomplete dual-arm validated command")
                    output = JointState()
                    output.header = copy.deepcopy(msg.header)  # Preserve deadline tag to the 1 kHz consumer.
                    output.name, output.position = list(contract.CONTROLLER_JOINT_NAMES), list(q)
                    outputs.append((side, output))
                for side, output in outputs:
                    validate_header(msg.header.frame_id, created, time.time_ns())
                    self.command_outputs[side].publish(output)
            except ValueError as error:
                self.fault = True
                self.get_logger().error(str(error))

    rclpy.init()
    node = Gateway() if args.role == "gateway" else Splitter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
