"""Weights-only checkpoint reader: no training workspace, optimizer, or dill execution."""

import hashlib
import sys
from pathlib import Path
import typing
from .config import ROOT

sys.path.insert(0, str(ROOT / "vendor"))


def _safe_type(name):
    return {
        "dict": dict,
        "list": list,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "NoneType": type(None),
    }[name]


def _plain_cache(*args):
    return {}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_checkpoint(path):
    import torch
    from omegaconf import OmegaConf, DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    allowed = [
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        typing.Any,
        dict,
        list,
        int,
        (_safe_type, "dill._dill._load_type"),
        (_plain_cache, "collections.defaultdict"),
    ]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    cfg = OmegaConf.to_container(payload["cfg"], resolve=False)
    # No arbitrary Hydra targets or resolvers from the file are executed.
    return cfg, payload["state_dicts"]


class Policy:
    def __init__(self, path, device="cuda:0"):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.policy.diffusion_unet_image_policy import (
            DiffusionUnetImagePolicy,
        )
        from .encoder import RGBDObsEncoder
        from .contract import OBS_SHAPES

        cfg, states = read_checkpoint(path)
        p = dict(cfg["policy"])
        p.pop("_target_")
        if (
            p["horizon"] != 16
            or p["n_obs_steps"] != 2
            or p["n_action_steps"] != 8
            or p["shape_meta"]["action"]["shape"] != [54]
        ):
            raise ValueError("Unsupported checkpoint temporal/action contract")
        if {
            k: tuple(v["shape"]) for k, v in p["shape_meta"]["obs"].items()
        } != OBS_SHAPES:
            raise ValueError("Unsupported checkpoint observation contract")
        scheduler = dict(p.pop("noise_scheduler"))
        target = scheduler.pop("_target_")
        if target != "diffusers.schedulers.scheduling_ddpm.DDPMScheduler":
            raise ValueError("Unexpected scheduler")
        enc = dict(p.pop("obs_encoder"))
        target = enc.pop("_target_")
        if target != "convest.dp.encoder.RGBDObsEncoder":
            raise ValueError("Unexpected encoder")
        self.model = DiffusionUnetImagePolicy(
            **p,
            noise_scheduler=DDPMScheduler(**scheduler),
            obs_encoder=RGBDObsEncoder(**enc),
        )
        key = "ema_model" if cfg["training"]["use_ema"] else "model"
        self.model.load_state_dict(states[key], strict=True)
        del states
        self.model.eval().requires_grad_(False).to(device)
        self.device = device
        self.cfg = cfg
        self.metadata = {
            "schema": "franka_wuji_dp_v1",
            "checkpoint": str(Path(path).resolve()),
            "sha256": digest(path),
            "weights": key,
            "horizon": 16,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "action_dim": 54,
            "order": ["left_arm_7", "right_arm_7", "left_hand_20", "right_hand_20"],
            "units": "radian",
            "action_semantics": "absolute_joint_target",
            "source_hz": 30,
            "num_inference_steps": p["num_inference_steps"],
            "warmed_up": False,
        }

    def infer(self, obs):
        import torch
        import numpy as np
        from .preprocess import validate

        obs = validate(obs)
        with torch.inference_mode():
            result = self.model.predict_action(
                {k: torch.from_numpy(v[None]).to(self.device) for k, v in obs.items()}
            )
        out = {k: v[0].cpu().numpy() for k, v in result.items()}
        if (
            out["action"].shape != (8, 54)
            or out["action_pred"].shape != (16, 54)
            or not all(np.isfinite(x).all() for x in out.values())
        ):
            raise ValueError("Invalid model output")
        return out
