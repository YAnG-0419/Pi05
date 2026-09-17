"""DP3 checkpoint contract for the existing dual-FR3/dual-Wuji bridge."""

import hashlib
from pathlib import Path

from fr3_wuji.paths import ROOT


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    checksum = ROOT / "MODEL.sha256"
    if not checkpoint.is_file():
        raise ValueError(f"DP3 checkpoint not found: {checkpoint}")
    if not checksum.is_file():
        raise ValueError(f"DP3 checksum file not found: {checksum}")
    actual = _sha256(checkpoint)
    expected = checksum.read_text(encoding="utf-8").split()[0]
    if actual != expected:
        raise ValueError(f"DP3 checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    return {
        "config": "dp3_bimanual_franka_wuji",
        "action_dim": 54,
        "action_horizon": 8,
        "action_sample_hz": 20,
        "observation_steps": 4,
        "point_count": 1024,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "action_representation": "absolute_joint_positions",
        "hardware_output": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual,
    }, checksum
