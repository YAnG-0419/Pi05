"""54D pi05 adapter retained from the original deployment repository."""

import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        raise ValueError("Missing complete JAX params directory")
    files = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(files) != 1:
        raise ValueError("Expected exactly one checkpoint-owned normalization file")
    stats = json.loads(files[0].read_text())["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[key][field], dtype=float)
            if values.shape != (54,) or not np.isfinite(values).all():
                raise ValueError(f"{key}.{field} must contain 54 finite values")
        if np.any(np.asarray(stats[key]["std"]) < 0):
            raise ValueError("Negative standard deviation")
        if np.any(np.asarray(stats[key]["q01"]) > stats[key]["q99"]):
            raise ValueError("Invalid quantile order")
    return {
        "config": "pi05_fr3_wuji",
        "action_dim": 54,
        "action_horizon": 50,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "hardware_output": False,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": hashlib.sha256(
            (checkpoint / "params/manifest.ocdbt").read_bytes()
        ).hexdigest(),
        "normalization_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "action_representation": "absolute_joint_positions_after_arm_delta_transform",
    }, files[0]


def create_policy(checkpoint, stats_file, prompt):
    from openpi.inference import checkpoints, config
    from openpi.policies import policy_config

    stats = checkpoints.load_norm_stats(
        checkpoint / "assets", str(stats_file.parent.relative_to(checkpoint / "assets"))
    )
    return policy_config.create_trained_policy(
        config.get_config("pi05_fr3_wuji"), checkpoint, norm_stats=stats, default_prompt=prompt
    )
