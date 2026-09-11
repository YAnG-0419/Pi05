"""Checkpoint-owned normalization and provenance for single-shot deployment."""

import argparse
from pathlib import Path

import numpy as np

from .qualification import digest


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    stats = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(stats) != 1:
        raise ValueError("Expected exactly one checkpoint-owned normalization file")
    return {
        "config": "pi05_fr3_wuji",
        "action_dim": 54,
        "action_horizon": 50,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "hardware_output": False,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": digest(checkpoint / "params/manifest.ocdbt"),
        "normalization_sha256": digest(stats[0]),
        "action_representation": "absolute_joint_positions_after_arm_delta_transform",
    }, stats[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/19999"))
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    metadata, stats = checkpoint_contract(args.checkpoint)
    if args.check_only:
        print(metadata)
        return
    import jax

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import checkpoints
    from openpi.training import config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU inference required")
    checkpoint = Path(metadata["checkpoint"])
    norm_stats = checkpoints.load_norm_stats(
        checkpoint / "assets", str(stats.parent.relative_to(checkpoint / "assets"))
    )
    policy = policy_config.create_trained_policy(config.get_config("pi05_fr3_wuji"), checkpoint, norm_stats=norm_stats)
    from experiments.weight_motion_eval.reference import deployment_module

    observation = deployment_module("observation")
    warmup = {"observation/state": np.zeros(54, dtype=np.float32), "prompt": observation.PROMPT}
    warmup.update(
        {
            key: np.zeros(observation.IMAGE_SHAPES[source], dtype=np.uint8)
            for source, key in observation.IMAGE_KEYS.items()
        }
    )
    # Warmup output is discarded; the motion client always requests a fresh scene.
    policy.infer(warmup)
    metadata["warmed_up"] = True
    print(f"Checkpoint ready at ws://127.0.0.1:{args.port}", flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
