"""Checkpoint-owned normalization and provenance for single-shot deployment."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from experiments.weight_motion_eval.oneshot.qualification import digest


def _checked_warmup_actions(result, label):
    if not isinstance(result, dict) or "actions" not in result:
        raise ValueError(f"{label} warmup did not return actions")
    actions = np.asarray(result["actions"])
    if actions.shape != (50, 54) or not np.isfinite(actions).all():
        raise ValueError(f"{label} warmup must return exactly 50 x 54 finite actions")
    return actions


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

    from openpi_client import rtc
    from openpi.policies import fr3_wuji_policy
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
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_fr3_wuji"),
        checkpoint,
        norm_stats=norm_stats,
        rtc_action_repack=fr3_wuji_policy.rtc_actions_to_dataset_order,
    )
    from experiments.weight_motion_eval.reference import deployment_module

    observation = deployment_module("observation")
    warmup = {"observation/state": np.zeros(54, dtype=np.float32), "prompt": observation.PROMPT}
    warmup.update(
        {
            key: np.zeros(observation.IMAGE_SHAPES[source], dtype=np.uint8)
            for source, key in observation.IMAGE_KEYS.items()
        }
    )
    # Compile both signatures before listening. The motion client always asks
    # for a fresh scene, so neither warmup prediction can reach hardware.
    vanilla = policy.infer(warmup)
    vanilla_actions = _checked_warmup_actions(vanilla, "Vanilla")
    print("Vanilla JAX inference warmup complete", flush=True)
    prefix = vanilla_actions[:25]
    source_sha256 = hashlib.sha256(prefix.astype("<f8").tobytes()).hexdigest()
    warmup_context = rtc.validate_context(
        {
            "protocol_version": rtc.PROTOCOL_VERSION,
            "token": "rtc-warmup",
            "source_number": 1,
            "source_sha256": source_sha256,
            "previous_actions": prefix,
            "inference_delay_steps": 5,
            "prefix_attention_horizon": 25,
            "prefix_attention_schedule": "exp",
            "max_guidance_weight": 5.0,
        }
    )
    guided = policy.infer(warmup, rtc_context=warmup_context)
    _checked_warmup_actions(guided, "RTC guided")
    rtc.validate_receipt(guided.get(rtc.RTC_RECEIPT_KEY), warmup_context)
    metadata.update(
        warmed_up=True,
        rtc_supported=True,
        rtc_protocol_version=rtc.PROTOCOL_VERSION,
        rtc_algorithm="soft_prefix_guidance",
        rtc_prefix_attention_schedules=sorted(rtc.PREFIX_ATTENTION_SCHEDULES),
        rtc_default_max_guidance_weight=5.0,
        rtc_warmed_up=True,
    )
    print(
        "RTC guided JAX inference warmup complete: "
        f"protocol={metadata['rtc_protocol_version']}, algorithm={metadata['rtc_algorithm']}, "
        f"schedules={metadata['rtc_prefix_attention_schedules']}",
        flush=True,
    )
    print(f"Checkpoint ready at ws://127.0.0.1:{args.port}", flush=True)
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
