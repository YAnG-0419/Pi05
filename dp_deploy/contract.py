"""DP uses LA7, RA7, LH20, RH20; pi0.5 device IPC uses LA7, LH20, RA7, RH20."""

import numpy as np

DP_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")
DEVICE_FROM_DP = np.r_[0:7, 14:34, 7:14, 34:54]
DP_FROM_DEVICE = np.argsort(DEVICE_FROM_DP)
RGB_KEYS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
OBS_SHAPES = {
    "joint_pos": (54,),
    **{k: (3, 240, 320) for k in RGB_KEYS},
    "head_depth": (1, 240, 320),
}


def to_device(x):
    x = np.asarray(x)
    if x.shape[-1] != 54:
        raise ValueError("Expected DP 54-vector")
    return x[..., DEVICE_FROM_DP].copy()


def to_dp(x):
    x = np.asarray(x)
    if x.shape[-1] != 54:
        raise ValueError("Expected device 54-vector")
    return x[..., DP_FROM_DEVICE].copy()


def joint_names(group):
    side, part = group.split("_")
    if part == "arm":
        return tuple(f"{side}_fr3_joint{i}" for i in range(1, 8))
    prefix = "l" if side == "left" else "r"
    suffix = ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"]
    for finger in ("index", "middle", "ring", "pinky"):
        stem = finger if finger == "pinky" else finger + "_finger"
        suffix.extend(f"{stem}_{j}" for j in ("mcp_flex", "mcp_abd", "pip", "dip"))
    return tuple(f"{prefix}_{j}" for j in suffix)


def ordered_positions(group, names, values):
    expected = joint_names(group)
    names = list(names)
    values = np.asarray(values)
    if (
        len(names) != len(set(names))
        or not set(expected) <= set(names)
        or values.shape != (len(names),)
    ):
        raise ValueError(f"{group}: missing/duplicate joints or wrong position shape")
    if group.endswith("hand") and set(names) != set(expected):
        raise ValueError("Unexpected hand mapping")
    result = values[[names.index(n) for n in expected]].astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite measured joints")
    return result
