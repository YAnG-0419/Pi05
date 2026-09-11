"""ROS feedback and ArmCommand source; never publishes the FR3 command bus."""

import time

import numpy as np

from .core import ARM
from .transport import ros_header


class RosArms:
    def __init__(self, *, publish_reset_idle=False, output_enabled=True):
        import rclpy
        from rclpy.qos import DurabilityPolicy
        from rclpy.qos import QoSProfile
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
        from teleop_core import contract
        from teleop_interfaces.msg import ArmCommand
        from teleop_interfaces.msg import ArmCommandStatus

        self.rclpy, self.contract = rclpy, contract
        self.JointState, self.ArmCommand = JointState, ArmCommand
        self.node = rclpy.create_node("pi05_oneshot_devices")
        self.samples, self.statuses = {}, {}
        self.offset = time.time() - time.monotonic()
        self.output = (
            self.node.create_publisher(ArmCommand, contract.SOURCE_COMMAND_TOPIC, 1) if output_enabled else None
        )
        self.reset_idle = None
        if publish_reset_idle:
            # The one-shot bringup omits the legacy reset server because it is
            # a second direct controller-command publisher. Reset is unavailable
            # in this session; competing reset-state publishers reject acquisition.
            self.reset_idle = self.node.create_publisher(
                Bool, contract.RESET_ACTIVE_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            )
            self.reset_idle.publish(Bool(data=False))
        self.hand_outputs = {
            side: self.node.create_publisher(JointState, contract.WUJI_STATE_TOPIC.format(side=side), 1)
            for side in ("left", "right")
        }
        for side in ("left", "right"):
            self.node.create_subscription(
                JointState,
                contract.ARM_STATE_TOPIC.format(side=side),
                lambda msg, side=side: self.sample(side, msg),
                qos_profile_sensor_data,
            )
        self.node.create_subscription(ArmCommandStatus, contract.COMMAND_STATUS_TOPIC, self.status, 10)
        self.stopped = False
        self.last_hand_publish = 0.0
        self.hand_feedback_errors = {}

    def sample(self, side, message):
        from teleop_core.joint_state import ordered_arm_positions

        now, wall = time.monotonic(), time.time()
        try:
            if len(message.name) != len(set(message.name)):
                raise ValueError("Duplicate measured arm joint names")
            if len(message.position) != len(message.name) or len(message.velocity) != len(message.name):
                raise ValueError("Full measured position and velocity are required")
            q = ordered_arm_positions(message.name, message.position, side)
            dq = ordered_arm_positions(message.name, message.velocity, side)
            age = wall - message.header.stamp.sec - message.header.stamp.nanosec / 1e9
            if not 0 <= age <= 0.15:
                raise ValueError("Stale or future measured arm state")
            self.samples[side] = (q, dq, now - age, now)
        except ValueError:
            self.samples.pop(side, None)

    def status(self, msg):
        if msg.source == "openpi":
            self.statuses[(msg.session_id, msg.sequence)] = msg
            while len(self.statuses) > 8:
                self.statuses.pop(next(iter(self.statuses)))

    def spin(self, budget=0.0002):
        end = time.monotonic() + budget
        for _ in range(128):
            self.rclpy.spin_once(self.node, timeout_sec=0)
            if time.monotonic() >= end:
                break

    def feedback(self, now):
        self.spin(0.0001)
        now = time.monotonic()
        if abs(time.time() - now - self.offset) > 0.05:
            raise ValueError("ROS device clock offset changed")
        if set(self.samples) != {"left", "right"}:
            raise ValueError("Waiting for both measured arms with velocities")
        samples = [self.samples[side] for side in ("left", "right")]
        if any(not 0 <= now - stamp <= 0.15 for sample in samples for stamp in sample[2:]):
            raise ValueError("Arm source or receipt feedback expired")
        return (
            np.concatenate([s[0] for s in samples]),
            np.concatenate([s[1] for s in samples]),
            tuple(s[2] for s in samples),
            tuple(s[3] for s in samples),
        )

    def check_ownership(self):
        if self.output is None:
            raise RuntimeError("Read-only ROS adapter cannot acquire arm output")
        c = self.contract
        expected = [c.SOURCE_COMMAND_TOPIC, c.ARM_COMMAND_TOPIC, c.RESET_ACTIVE_TOPIC]
        expected += [c.CONTROLLER_COMMAND_TOPIC.format(side=s) for s in ("left", "right")]
        expected += [c.WUJI_STATE_TOPIC.format(side=s) for s in ("left", "right")]
        if any(self.node.count_publishers(topic) != 1 for topic in expected):
            raise RuntimeError("Missing or competing gateway, splitter, policy or hand owner")
        if any(self.node.count_subscribers(c.CONTROLLER_COMMAND_TOPIC.format(side=s)) != 1 for s in ("left", "right")):
            raise RuntimeError("Expected one final arm controller on each side")

    def publish_hands(self, hands):
        now = time.monotonic()
        publish_now = now - self.last_hand_publish >= 0.01
        for side, hand in hands.items():
            try:
                q, dq, stamp = hand.poll(time.monotonic(), time.time())
            except ValueError as error:
                self.hand_feedback_errors[side] = str(error)
                continue  # Do not publish missing or stale values with a new stamp.
            self.hand_feedback_errors.pop(side, None)
            if not publish_now:
                continue
            message = self.JointState()
            wall_stamp = int((time.time() - (time.monotonic() - stamp)) * 1e9)
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(wall_stamp, 1_000_000_000)
            message.name = list(
                self.contract.WUJI_LEFT_JOINT_NAMES if side == "left" else self.contract.WUJI_RIGHT_JOINT_NAMES
            )
            message.position, message.velocity = q.tolist(), dq.tolist()
            self.hand_outputs[side].publish(message)
        if publish_now:
            self.last_hand_publish = now

    def submit(self, frame):
        if self.stopped or self.output is None:
            raise RuntimeError("Arm source cannot restart after stop")
        stamp, tag = ros_header(frame, time.monotonic(), time.time_ns())
        msg = self.ArmCommand()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        msg.header.frame_id = tag
        msg.source, msg.session_id, msg.sequence = "openpi", frame.run_id, frame.sequence
        msg.active_sides = ["left", "right"]
        msg.joint_names = list(self.contract.COMMAND_JOINT_NAMES)
        msg.positions = frame.positions[ARM].tolist()
        if time.monotonic() >= frame.valid_until:
            raise ValueError("Arm frame expired before ROS publication")
        self.output.publish(msg)
        key = (frame.run_id, frame.sequence)
        while time.monotonic() < frame.valid_until:
            self.spin(0.0003)
            if key in self.statuses:
                status = self.statuses.pop(key)
                if status.faults or list(status.accepted_sides) != ["left", "right"]:
                    raise RuntimeError("Gateway rejected frame: " + str(status.faults))
                if status.header != msg.header:
                    raise RuntimeError("Gateway acknowledgement changed the command deadline")
                return
        raise TimeoutError("Gateway acknowledgement missed original frame deadline")

    def stop(self):
        self.stopped = True
        # No direct bus/controller writes. The overlay latches a hold when the
        # last accepted frame expires (at most 20 ms after frame creation).

    def close(self):
        self.node.destroy_node()
