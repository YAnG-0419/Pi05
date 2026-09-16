"""Synthetic ROS publisher + actual collector, in an isolated network namespace.

This creates only sensor_msgs observations, never command messages or SDK objects.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from dp_deploy.config import load_config
from dp_deploy.contract import DP_GROUPS, RGB_KEYS, joint_names
from dp_deploy.wire import request

c = load_config()
rclpy.init()
node = rclpy.create_node("dp_synthetic_observations")
publishers = {
    k: node.create_publisher(
        JointState if k in DP_GROUPS else Image, t, qos_profile_sensor_data
    )
    for k, t in c["topics"].items()
}
child = subprocess.Popen([sys.executable, "-m", "dp_deploy.observe", "--port", "8876"])
start = time.monotonic()
offsets = {"left_arm": 0, "right_arm": 7, "left_hand": 14, "right_hand": 34}
result = None
try:
    while time.monotonic() - start < 12:
        if child.poll() is not None:
            raise RuntimeError("Collector process exited")
        stamp = node.get_clock().now().to_msg()
        for key, pub in publishers.items():
            if key in DP_GROUPS:
                m = JointState()
                m.name = list(joint_names(key))
                m.position = (
                    (np.arange(len(m.name)) + offsets[key]).astype(float).tolist()
                )
            else:
                m = Image()
                m.height = 240
                m.width = 320
                m.is_bigendian = False
                image = (
                    np.full((240, 320), 1000, np.uint16)
                    if key == "head_depth"
                    else np.full((240, 320, 3), 128, np.uint8)
                )
                m.encoding = "16UC1" if key == "head_depth" else "rgb8"
                m.step = 640 if key == "head_depth" else 960
                m.data = image.tobytes()
            m.header.stamp = stamp
            pub.publish(m)
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(1 / 30)
        if time.monotonic() - start > 0.5:
            try:
                result = request("http://127.0.0.1:8876/observation", timeout=0.15)
            except Exception:
                continue
            np.testing.assert_array_equal(
                result["joint_pos"], np.tile(np.arange(54), (2, 1))
            )
            assert (
                result["head_depth"].shape == (2, 240, 320)
                and result["head_depth"].dtype == np.uint16
            )
            assert all(result[k].shape == (2, 240, 320, 3) for k in RGB_KEYS)
            assert (
                0 <= time.time() - result["timestamps"][-1] < c["max_observation_age_s"]
            )
            break
    if result is None:
        raise TimeoutError("Synthetic ROS collection timed out")
    report = {
        "passed": True,
        "hardware_connected": False,
        "synthetic_observations": True,
        "state_shape": list(result["joint_pos"].shape),
        "depth_shape": list(result["head_depth"].shape),
    }
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
finally:
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
