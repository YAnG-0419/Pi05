"""DP3 inference server. It never imports or writes robot hardware."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


OBS_STEPS = 4
POINTS = 1024
STATE_DIM = 54
ACTION_STEPS = 8


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DP3Policy:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def infer(self, observation):
        import torch

        expected = {
            "point_cloud": (OBS_STEPS, POINTS, 3),
            "agent_pos": (OBS_STEPS, STATE_DIM),
        }
        arrays = {}
        for key, shape in expected.items():
            value = np.array(observation.get(key), dtype=np.float32, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite {shape}, got {value.shape}")
            arrays[key] = torch.from_numpy(value[None]).to(self.device)
        with torch.inference_mode():
            actions = self.policy.predict_action(arrays)["action"][0].detach().cpu().numpy()
        if actions.shape != (ACTION_STEPS, STATE_DIM) or not np.isfinite(actions).all():
            raise ValueError(f"DP3 returned invalid actions: {actions.shape}")
        return {"actions": actions.astype(np.float32, copy=False)}


def load_policy(checkpoint, project, device):
    sys.path.insert(0, str(project))
    import dill
    import hydra
    from omegaconf import OmegaConf
    import torch

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["cfg"]
    if (int(cfg.n_obs_steps), int(cfg.n_action_steps)) != (OBS_STEPS, ACTION_STEPS):
        raise ValueError(
            f"Checkpoint contract is obs={cfg.n_obs_steps}, actions={cfg.n_action_steps}; "
            f"expected {OBS_STEPS}/{ACTION_STEPS}"
        )
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    del payload
    gc.collect()
    policy.to(device).eval()
    return DP3Policy(policy, device), state_key


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "packages/openpi-client/src")]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dp3-project",
        type=Path,
        default=Path(os.environ.get("DP3_PROJECT", root / "dp3_runtime")),
    )
    parser.add_argument("--sha256-file", type=Path, default=root / "MODEL.sha256")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    project = args.dp3_project.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if not (project / "diffusion_policy_3d/policy/dp3.py").is_file():
        parser.error(f"DP3 source not found: {project}")
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")

    checkpoint_sha256 = sha256(checkpoint)
    if args.sha256_file:
        checksum_file = args.sha256_file.expanduser().resolve()
        if not checksum_file.is_file():
            parser.error(f"checksum file not found: {checksum_file}")
        expected_sha256 = checksum_file.read_text(encoding="utf-8").split()[0]
        if checkpoint_sha256 != expected_sha256:
            parser.error(
                f"checkpoint SHA256 mismatch: expected {expected_sha256}, got {checkpoint_sha256}"
            )
    metadata = {
        "config": "dp3_bimanual_franka_wuji",
        "action_dim": STATE_DIM,
        "action_horizon": ACTION_STEPS,
        "action_sample_hz": 20,
        "observation_steps": OBS_STEPS,
        "point_count": POINTS,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "action_representation": "absolute_joint_positions",
        "hardware_output": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
    }
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    metadata["torch"] = torch.__version__
    metadata["device"] = str(device)
    if args.check_only:
        sys.path.insert(0, str(project))
        import diffusers  # noqa: F401
        import dill  # noqa: F401
        import einops  # noqa: F401
        import hydra  # noqa: F401
        import msgpack  # noqa: F401
        import websockets  # noqa: F401
        import zarr  # noqa: F401
        from diffusion_policy_3d.policy.dp3 import DP3  # noqa: F401
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: F401

        print(json.dumps(metadata, indent=2))
        return

    policy, weights = load_policy(checkpoint, project, device)
    metadata["weights"] = weights
    synthetic = {
        "point_cloud": np.zeros((OBS_STEPS, POINTS, 3), dtype=np.float32),
        "agent_pos": np.zeros((OBS_STEPS, STATE_DIM), dtype=np.float32),
    }
    actions = policy.infer(synthetic)["actions"]
    metadata["warmed_up"] = True
    print(f"DP3 warmup OK: {actions.shape} on {device}", flush=True)
    if args.smoke_only:
        return

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    print(f"Serving DP3 at ws://{args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
