"""Exact training letterbox/AREA RGB and nearest-centre depth preprocessing."""

import math
import numpy as np
from .contract import OBS_SHAPES, RGB_KEYS


def resize(array, width=320, height=240, depth=False):
    h, w = array.shape[:2]
    ph, pw = (
        max(h, math.ceil(w * height / width)),
        max(w, math.ceil(h * width / height)),
    )
    top, left = (ph - h) // 2, (pw - w) // 2
    padding = ((top, ph - h - top), (left, pw - w - left)) + (
        () if depth else ((0, 0),)
    )
    array = np.pad(array, padding)
    if depth:
        ys = np.minimum(((np.arange(height) + 0.5) * ph / height).astype(int), ph - 1)
        xs = np.minimum(((np.arange(width) + 0.5) * pw / width).astype(int), pw - 1)
        return array[ys[:, None], xs[None, :], None]
    import av

    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
    return frame.reformat(
        width=width, height=height, format="rgb24", interpolation="AREA"
    ).to_ndarray(format="rgb24")


def prepare(raw, config):
    obs = {"joint_pos": np.asarray(raw["joint_pos"], dtype=np.float32)}
    for key in RGB_KEYS:
        x = np.asarray(raw[key])
        if x.dtype != np.uint8 or x.ndim != 4 or x.shape[0] != 2 or x.shape[-1] != 3:
            raise ValueError(f"{key}: expected two RGB uint8 HWC frames")
        if any(not np.any(f) for f in x):
            raise ValueError(f"{key}: black frame")
        obs[key] = (
            np.stack([resize(f).transpose(2, 0, 1) for f in x]).astype(np.float32) / 255
        )
    x = np.asarray(raw["head_depth"])
    if (
        x.ndim != 3
        or x.shape[0] != 2
        or x.dtype not in (np.uint16, np.float32, np.float64)
    ):
        raise ValueError("head_depth: expected two uint16 mm or float metres images")
    values = x.astype(np.float32)
    if x.dtype == np.uint16:
        values *= config["depth_scale"]
    values[~np.isfinite(values) | (values <= 0)] = 0
    if any(not np.any(f) for f in values):
        raise ValueError("Depth has no valid pixels")
    obs["head_depth"] = np.stack(
        [resize(f, depth=True).transpose(2, 0, 1) for f in values]
    )
    obs["head_depth"] = (
        np.clip(obs["head_depth"], 0, config["depth_max_m"]) / config["depth_max_m"]
    )
    return validate(obs)


def validate(obs):
    if set(obs) != set(OBS_SHAPES):
        raise ValueError("Observation keys differ from checkpoint")
    for k, shape in OBS_SHAPES.items():
        x = np.asarray(obs[k])
        if x.shape != (2, *shape) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(
                f"{k}: require finite float32 {(2, *shape)}, got {x.shape}/{x.dtype}"
            )
        if k != "joint_pos" and (x.min() < 0 or x.max() > 1):
            raise ValueError(f"{k}: expected [0,1]")
    return {k: np.ascontiguousarray(v) for k, v in obs.items()}
