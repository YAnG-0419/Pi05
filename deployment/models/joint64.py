"""Original 64D pi05 checkpoint adapter with a 54D hardware contract."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np


def check_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not (path / "params/manifest.ocdbt").is_file():
        raise ValueError("Missing complete JAX params directory")
    files = list((path / "assets").rglob("norm_stats.json"))
    if len(files) != 1:
        raise ValueError("Expected exactly one checkpoint-owned norm_stats.json")
    stats = json.loads(files[0].read_text())["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[key][field], dtype=float)
            if values.shape != (64,) or not np.isfinite(values).all():
                raise ValueError(f"{key}.{field} must contain 64 finite values")
            if np.any(values[54:] != 0):
                raise ValueError(f"{key}.{field}: padding dimensions must be zero")
        if np.any(np.asarray(stats[key]["std"]) < 0):
            raise ValueError("Negative standard deviation")
        if np.any(np.asarray(stats[key]["q01"]) > stats[key]["q99"]):
            raise ValueError("Invalid quantile order")
    return path, files[0]


@dataclasses.dataclass(frozen=True)
class ModelOrderedState:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (54,) or not np.isfinite(state).all():
            raise ValueError("Expected one finite 54D physical state")
        ordered = np.concatenate((state[:7], state[27:34], state[7:27], state[34:54]))
        return {**data, "state": np.pad(ordered, (0, 10))}


@dataclasses.dataclass(frozen=True)
class PhysicalAbsoluteActions:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.shape != (50, 64) or not np.isfinite(actions).all():
            raise ValueError("Expected finite model actions of shape (50, 64)")
        physical = np.concatenate(
            (actions[:, :7], actions[:, 14:34], actions[:, 7:14], actions[:, 34:54]), axis=-1
        )
        return {"actions": physical}


@dataclasses.dataclass(frozen=True)
class AdaptData:
    base: object

    def create(self, assets_dirs, model_config):
        from openpi import transforms

        data = dataclasses.replace(self.base, extra_delta_transform=False).create(assets_dirs, model_config)
        group = transforms.Group(
            inputs=(*data.data_transforms.inputs, ModelOrderedState()),
            outputs=(PhysicalAbsoluteActions(),),
        )
        return dataclasses.replace(data, data_transforms=group)


class CheckedPolicy:
    def __init__(self, policy, metadata):
        self.policy, self.metadata = policy, metadata

    def infer(self, obs, **kwargs):
        state = np.asarray(obs["observation/state"])
        if state.shape != (54,) or not np.isfinite(state).all():
            raise ValueError("Expected finite observation/state of shape (54,)")
        result = self.policy.infer(obs, **kwargs)
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 54) or not np.isfinite(actions).all():
            raise ValueError("Expected finite physical actions of shape (50, 54)")
        return result


def checkpoint_contract(checkpoint):
    checkpoint, stats_file = check_checkpoint(checkpoint)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "config": "pi05_fr3_wuji_30000_64to54",
        "adapter_revision": 2,
        "model_state_order": ["left_arm_7", "right_arm_7", "left_hand_20", "right_hand_20", "padding_10"],
        "model_action_order": ["left_arm_7", "right_arm_7", "left_hand_20", "right_hand_20", "padding_10"],
        "model_action_dim": 64,
        "action_dim": 54,
        "action_horizon": 50,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "checkpoint": str(checkpoint),
        "hardware_output": False,
        "padding_dim": 10,
        "checkpoint_manifest_sha256": digest(checkpoint / "params/manifest.ocdbt"),
        "normalization_sha256": digest(stats_file),
        "action_representation": "absolute_joint_positions",
    }, stats_file


def create_policy(checkpoint, stats_file, prompt):
    from openpi.inference import checkpoints, config
    from openpi.policies import policy_config

    base = config.get_config("pi05_fr3_wuji")
    adapted = dataclasses.replace(
        base,
        name="pi05_fr3_wuji_30000_64to54",
        model=dataclasses.replace(base.model, action_dim=64),
        data=AdaptData(base.data),
    )
    stats = checkpoints.load_norm_stats(
        checkpoint / "assets", str(stats_file.parent.relative_to(checkpoint / "assets"))
    )
    policy = policy_config.create_trained_policy(adapted, checkpoint, norm_stats=stats, default_prompt=prompt)
    metadata, _ = checkpoint_contract(checkpoint)
    return CheckedPolicy(policy, metadata)
