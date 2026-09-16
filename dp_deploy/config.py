"""Configuration shared by model host and ROS/SDK containers."""

import json
import os
from pathlib import Path
import math
import ipaddress
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(
        path or os.environ.get("DP_DEPLOY_CONFIG", ROOT / "configs/deploy.json")
    ).resolve()
    c = json.loads(path.read_text())
    for key, env in {
        "checkpoint": "DP_CHECKPOINT",
        "reference": "DP_REFERENCE",
        "sdk_site": "DP_SDK_SITE",
        "docker_image": "DP_DOCKER_IMAGE",
    }.items():
        if os.environ.get(env):
            c[key] = os.environ[env]
    for key in ("checkpoint", "reference", "sdk_site"):
        c[key] = str(Path(c[key]).expanduser().resolve())
    for k in (
        "depth_scale",
        "depth_max_m",
        "source_hz",
        "control_hz",
        "arm_speed_rad_s",
        "target_arm_speed_rad_s",
        "max_initial_arm_delta_rad",
        "max_initial_hand_delta_rad",
        "inference_timeout_s",
        "observation_timeout_s",
        "max_observation_age_s",
        "max_state_age_s",
        "max_image_age_s",
        "max_hold_s",
        "max_run_s",
    ):
        if isinstance(c[k], bool) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f"{k} must be finite and positive")
    if c["source_hz"] != 30 or c["control_hz"] != 100:
        raise ValueError("This deployment requires training 30 Hz / device 100 Hz")
    if not c["target_arm_speed_rad_s"] < c["arm_speed_rad_s"] <= 1:
        raise ValueError("Require target arm speed < device speed <= 1 rad/s")
    if c["depth_max_m"] != 2.0:
        raise ValueError("This checkpoint was trained with depth_max_m=2.0")
    for side in ("left", "right"):
        ipaddress.IPv4Address(c[side + "_arm_ip"])
        address, port = c[side + "_hand_address"].rsplit(":", 1)
        ipaddress.IPv4Address(address)
        if not 1 <= int(port) <= 65535:
            raise ValueError("Invalid hand port")
    if (
        c["left_arm_ip"] == c["right_arm_ip"]
        or c["left_hand_address"] == c["right_hand_address"]
    ):
        raise ValueError("Left/right device addresses must differ")
    for key in ("policy_url", "observation_url"):
        u = urlparse(c[key])
        if (
            u.scheme != "http"
            or not u.hostname
            or u.port is None
            or u.path not in ("", "/")
        ):
            raise ValueError(f"{key}: expected http://host:port")
    if urlparse(c["observation_url"]).hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Observation collector runs on this host")
    if type(c["ros_domain_id"]) is not int or not 0 <= c["ros_domain_id"] <= 232:
        raise ValueError("Invalid ROS domain ID")
    if c["ros_localhost_only"] not in (0, 1):
        raise ValueError("ros_localhost_only must be 0 or 1")
    return c
