"""Hardware-free GR00T/FR3 wire and joint-order contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

PROTOCOL = "pi05.groot-fr3/v1"
ACTION_HZ = 30
ACTION_HORIZON = 16
ACTION_DIM = 54
MAX_BODY_BYTES = 32 * 1024 * 1024

PHYSICAL_GROUPS = ("left_arm", "left_hand", "right_arm", "right_hand")
PHYSICAL_SLICES = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 27),
    "right_arm": slice(27, 34),
    "right_hand": slice(34, 54),
}
GROOT_GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")
GROOT_SLICES = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_hand": slice(14, 34),
    "right_hand": slice(34, 54),
}
IMAGE_KEYS = {
    "head": "observation/image",
    "left_wrist": "observation/left_wrist_image",
    "right_wrist": "observation/right_wrist_image",
}
ACTION_ORDER = ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"]
_SHA256 = re.compile(r"[0-9a-f]{64}")


def finite_vector(value, *, label: str = "state") -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain exactly {ACTION_DIM} finite values")
    return result


def finite_actions(value, *, order: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(result).all():
        raise ValueError(
            f"{order} actions must have shape ({ACTION_HORIZON}, {ACTION_DIM}) and contain only finite values"
        )
    return result


def split_physical_state(value) -> dict[str, np.ndarray]:
    state = finite_vector(value)
    return {name: state[index].copy() for name, index in PHYSICAL_SLICES.items()}


def physical_to_groot(value) -> np.ndarray:
    groups = split_physical_state(value)
    return np.concatenate([groups[name] for name in GROOT_GROUPS])


def groot_to_physical(value) -> np.ndarray:
    actions = finite_actions(value, order="GR00T")
    groups = {name: actions[:, index] for name, index in GROOT_SLICES.items()}
    return np.concatenate([groups[name] for name in PHYSICAL_GROUPS], axis=-1)


def validate_images(images: dict[str, object]) -> dict[str, np.ndarray]:
    result = {}
    for name in IMAGE_KEYS:
        if name not in images:
            raise ValueError(f"Missing {name} image")
        image = np.asarray(images[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} must be RGB uint8 [height,width,3]")
        result[name] = image
    return result


def semantic_json_sha256(path: Path) -> str:
    value = json.loads(path.read_text())
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(checkpoint: Path) -> tuple[str, list[dict[str, object]]]:
    checkpoint = checkpoint.resolve()
    required = (
        "config.json",
        "embodiment_id.json",
        "processor_config.json",
        "statistics.json",
        "inference_recipe.json",
        "reload_verification.json",
    )
    files = [checkpoint / name for name in required]
    files.extend(sorted(checkpoint.glob("model*.safetensors*")))
    missing = [str(path) for path in files[: len(required)] if not path.is_file()]
    if missing:
        raise ValueError("Incomplete GR00T checkpoint: " + ", ".join(missing))
    if not files[len(required) :]:
        raise ValueError("GR00T checkpoint has no safetensors weights")
    manifest = []
    for path in files:
        if not path.is_file():
            continue
        manifest.append(
            {
                "path": str(path.relative_to(checkpoint)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), manifest


def validate_metadata(metadata: object, *, expected_model_sha256: str | None = None) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("GR00T server metadata must be a dictionary")
    expected = {
        "protocol": PROTOCOL,
        "backend": "groot_n1d7",
        "action_shape": [ACTION_HORIZON, ACTION_DIM],
        "action_hz": ACTION_HZ,
        "action_order": ACTION_ORDER,
        "joint_units": "radians",
        "action_semantics": "absolute_joint_targets",
        "reload_verified": True,
        "warmed_up": True,
        "hardware_output": False,
    }
    wrong = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if wrong:
        raise ValueError(f"Incompatible GR00T server metadata: {wrong}")
    for key in ("model_manifest_sha256", "statistics_sha256"):
        value = metadata.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"Invalid {key} in GR00T server metadata")
    if expected_model_sha256 is not None:
        if _SHA256.fullmatch(expected_model_sha256) is None:
            raise ValueError("Expected model SHA256 must contain 64 lowercase hexadecimal characters")
        if metadata["model_manifest_sha256"] != expected_model_sha256:
            raise ValueError("Connected GR00T server does not match the explicitly admitted checkpoint")
    return dict(metadata)
