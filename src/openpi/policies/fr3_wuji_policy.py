"""Policy transforms for an FR3 dual-arm robot with Wuji hands."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

FR3_WUJI_ACTION_DIM = 54


def make_fr3_wuji_example() -> dict:
    """Creates an inference example in the FR3/Wuji policy input format."""
    return {
        "observation/state": np.random.rand(FR3_WUJI_ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(400, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "perform the task with both hands",
    }


def _parse_image(image) -> np.ndarray:
    """Converts an image to uint8 HWC format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _reorder_state(state) -> np.ndarray:
    state = np.asarray(state)
    if state.shape[-1] == FR3_WUJI_ACTION_DIM:
        return state
    if state.shape[-1] != 108:
        raise ValueError(f"Expected FR3/Wuji state dimension 54 or 108, got {state.shape[-1]}")
    return np.concatenate(
        [state[..., 0:7], state[..., 28:48], state[..., 14:21], state[..., 68:88]],
        axis=-1,
    )


def _reorder_actions(actions) -> np.ndarray:
    actions = np.asarray(actions)
    if actions.shape[-1] != FR3_WUJI_ACTION_DIM:
        raise ValueError(f"Expected FR3/Wuji action dimension 54, got {actions.shape[-1]}")
    return np.concatenate(
        [actions[..., 0:7], actions[..., 14:34], actions[..., 7:14], actions[..., 34:54]],
        axis=-1,
    )


@dataclasses.dataclass(frozen=True)
class Fr3WujiInputs(transforms.DataTransformFn):
    """Converts raw FR3/Wuji observations and actions to model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": _reorder_state(data["observation/state"]),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _reorder_actions(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Fr3WujiOutputs(transforms.DataTransformFn):
    """Returns actions in [left arm, left hand, right arm, right hand] order."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :FR3_WUJI_ACTION_DIM]}
