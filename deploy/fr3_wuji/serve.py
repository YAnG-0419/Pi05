"""Local FR3/Wuji inference only. No ROS publishers or hardware connections."""

import argparse
import json
import logging
import math
from pathlib import Path

CONFIG = "pi05_fr3_wuji"
PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand "
    "and place it in the left basket."
)


def check_checkpoint(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    params = checkpoint / "params"
    if not params.is_dir() or not any(params.iterdir()):
        raise ValueError(f"Missing/non-populated params directory: {params}. Copy the full JAX step directory.")
    stats_path = checkpoint / "assets/fr3_wuji/tomato/norm_stats.json"
    with stats_path.open() as stream:
        stats = json.load(stream)["norm_stats"]
    for key in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = stats[key][field]
            if (
                not isinstance(values, list)
                or len(values) != 54
                or not all(isinstance(value, int | float) and math.isfinite(value) for value in values)
            ):
                raise ValueError(f"Expected 54 finite values at {stats_path}: {key}.{field}")
        if any(value < 0 for value in stats[key]["std"]):
            raise ValueError(f"Negative standard deviation: {key}")
        if any(low > high for low, high in zip(stats[key]["q01"], stats[key]["q99"], strict=True)):
            raise ValueError(f"Invalid quantile order: {key}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Step directory containing params/ and assets/.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--check-only", action="store_true", help="Check layout/stats without importing JAX or loading weights."
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        checkpoint = check_checkpoint(args.checkpoint)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Checkpoint layout/stats OK: {checkpoint}", flush=True)
    if args.check_only:
        print("Parameter completeness and compatibility still require actual model loading.")
        return

    import jax

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"CUDA GPU required; found {devices}")
    logging.info("JAX devices: %s", devices)
    policy = policy_config.create_trained_policy(config.get_config(CONFIG), checkpoint, default_prompt=args.prompt)
    metadata = {
        **policy.metadata,
        "config": CONFIG,
        "action_dim": 54,
        "action_horizon": 50,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "hardware_output": False,
    }
    logging.info("Serving inference at ws://%s:%s (Ctrl-C to stop)", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
