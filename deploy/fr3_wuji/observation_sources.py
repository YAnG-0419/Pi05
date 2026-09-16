"""Isolated read-only producers: ROS/cameras+libfranka in Docker, SDK in Conda."""

import argparse
import contextlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
from observation import joint_names
from observation import send_record


def interrupted(*_):
    raise KeyboardInterrupt


def stop(process):
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def hands(args, sock):
    import wuji_sdk

    manager = wuji_sdk.SdkManager.instance()
    devices, subscriptions, last_sent = {}, {}, {}
    try:
        for side, address in (("left", args.left_hand), ("right", args.right_hand)):
            device = manager.connect(
                address=address,
                device_name="pi05_observe_" + side,
                options=wuji_sdk.ConnectOptions(enable_bridge=False, auto_time_sync_interval_ms=None),
            )
            devices[side] = device
            if str(device.handedness().get()).lower() != side or int(device.online_joints_count().get()) != 20:
                raise RuntimeError(f"{side}: wrong hand identity or incomplete joints")
            subscriptions[side] = device.joint_states().subscribe()
            last_sent[side] = 0
        while True:
            received = False
            for side, subscription in subscriptions.items():
                frame = subscription.recv()
                if frame is None:
                    continue
                received = True
                stamp = int(frame.header.timestamp_us)
                if stamp < last_sent[side]:
                    raise RuntimeError(f"{side}: device clock moved backwards")
                if stamp < last_sent[side] + 10_000:
                    continue
                last_sent[side] = stamp
                entries = list(frame.joints)
                expected = [finger * 5 + joint + 1 for finger in range(5) for joint in range(4)]
                by_id = {int(joint.nid): float(joint.position) for joint in entries}
                if len(entries) != 20 or int(frame.num_joints) != 20 or set(by_id) != set(expected):
                    raise RuntimeError(f"{side}: invalid firmware joint IDs")
                source = side + "_hand"
                send_record(
                    sock,
                    {
                        "source": source,
                        "stamp": stamp / 1e6,
                        "stamp_basis": "wuji_sdk_synced_host",
                        "units": "radian",
                        "measured": True,
                        "names": joint_names(source),
                        "positions": [by_id[nid] for nid in expected],
                    },
                )
            if not received:
                time.sleep(0.0001)
    finally:
        for subscription in subscriptions.values():
            subscription.close()
        for device in devices.values():
            device.disconnect()


def camera_images(args, sock):
    from cv_bridge import CvBridge
    import rclpy
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("pi05_observation_cam" + str(args.camera))
    bridge = CvBridge()

    def callback(message):
        received_wall = time.time()
        rgb = np.ascontiguousarray(bridge.imgmsg_to_cv2(message, desired_encoding="rgb8"))
        send_record(
            sock,
            {
                "source": f"cam{args.camera}",
                "stamp": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                "stamp_basis": "camera_global_ros_header",
                "producer_received_wall": received_wall,
                "shape": list(rgb.shape),
                "encoding": "rgb8",
            },
            rgb.tobytes(),
        )

    # Match record_gello.yaml: best-effort large images drop on this workcell's Cyclone DDS.
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    subscription = node.create_subscription(Image, f"/cam{args.camera}/color/image_raw", callback, qos)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def ros_and_arms(args, sock):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from teleop_core import contract

    rclpy.init()
    node = rclpy.create_node("pi05_readonly_observations")
    lock, refs, errors = threading.Lock(), [], []
    image_readers, image_logs = [], []
    camera = arm = None
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    camera_log = (logs / "cameras.txt").open("w")
    arm_log = (logs / "arms.txt").open("w")

    def joint_callback(source, message):
        send_record(
            sock,
            {
                "source": source,
                "stamp": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                "stamp_basis": "ros_joint_header",
                "names": list(message.name),
                "positions": list(message.position),
                "units": "radian",
                "measured": True,
            },
            lock=lock,
        )

    def read_arms():
        try:
            for line in arm.stdout:
                send_record(sock, json.loads(line), lock=lock)
            errors.append("Direct arm reader exited")
        except Exception as error:
            errors.append(str(error))

    try:
        for i in range(3):
            stream = (logs / f"cam{i}-reader.txt").open("w")
            image_logs.append(stream)
            image_readers.append(
                subprocess.Popen(
                    [sys.executable, __file__, "camera", "--camera", str(i), "--socket", args.socket],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            )
        for side in ("left", "right"):
            for part, mode, topic in (
                ("arm", args.arm_source, contract.ARM_STATE_TOPIC),
                ("hand", args.hand_source, contract.WUJI_STATE_TOPIC),
            ):
                if mode == "ros":
                    refs.append(
                        node.create_subscription(
                            JointState,
                            topic.format(side=side),
                            lambda message, source=side + "_" + part: joint_callback(source, message),
                            qos_profile_sensor_data,
                        )
                    )
        if args.start_cameras:
            camera = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "teleop_camera_bringup",
                    "triple_camera.launch.py",
                    "deployment_config:=" + args.camera_config,
                ],
                stdout=camera_log,
                stderr=subprocess.STDOUT,
            )
        if args.arm_source == "direct":
            arm = subprocess.Popen(
                ["/build/read_arms", args.left_arm, args.right_arm], stdout=subprocess.PIPE, stderr=arm_log, text=True
            )
            threading.Thread(target=read_arms, daemon=True).start()
        while not errors and rclpy.ok():
            if any(reader.poll() is not None for reader in image_readers):
                raise RuntimeError("Camera reader exited")
            if camera is not None and camera.poll() is not None:
                raise RuntimeError("Camera launch exited")
            rclpy.spin_once(node, timeout_sec=0.1)
        if errors:
            raise RuntimeError(errors[0])
    finally:
        stop(arm)
        stop(camera)
        for reader in image_readers:
            stop(reader)
        for stream in image_logs:
            stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        camera_log.close()
        arm_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("hands", "ros", "camera"))
    parser.add_argument("--camera", type=int, choices=(0, 1, 2))
    parser.add_argument("--socket", required=True)
    parser.add_argument("--logs", default="/logs")
    parser.add_argument("--arm-source", choices=("ros", "direct"), default="ros")
    parser.add_argument("--hand-source", choices=("ros", "sdk"), default="ros")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument(
        "--camera-config", default="/workspace/franka_upper_body_teleop/data_collection/config/cameras.yaml"
    )
    parser.add_argument("--left-arm", default="172.16.0.2")
    parser.add_argument("--right-arm", default="172.16.1.2")
    parser.add_argument("--left-hand", default="192.168.1.110:7447")
    parser.add_argument("--right-hand", default="192.168.2.111:7447")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(args.socket)
        with contextlib.suppress(KeyboardInterrupt):
            {"hands": hands, "ros": ros_and_arms, "camera": camera_images}[args.kind](args, sock)


if __name__ == "__main__":
    main()
