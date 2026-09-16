"""Local deployment dependencies; imports never connect to hardware."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = Path(os.environ.get(
    "OPENPI_TELEOP_ROOT", "/home/descfly/llx/gello_upper_body_teleop"
)).expanduser().resolve()
TELEOP_PYTHON = Path(os.environ.get(
    "OPENPI_TELEOP_PYTHON", "/home/descfly/miniforge3/envs/gello-upper-body-teleop/bin/python"
)).expanduser().absolute()
WUJI_SDK_SITE = Path(os.environ.get(
    "OPENPI_WUJI_SDK_SITE",
    str(TELEOP_PYTHON.parent.parent / "lib/python3.10/site-packages"),
)).expanduser().resolve()
CONTROL_IMAGE = os.environ.get("OPENPI_CONTROL_IMAGE", "franka-upper-body-teleop:latest")

CAMERA_CONFIG = Path(os.environ.get(
    "OPENPI_CAMERA_CONFIG", str(TELEOP_ROOT / "data_collection/config/cameras.yaml")
)).expanduser().resolve()
