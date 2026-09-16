"""Immutable, validated messages shared by replay, shadow and device adapters."""

from dataclasses import dataclass
import math

import numpy as np

GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 27), "right_arm": slice(27, 34), "right_hand": slice(34, 54)}
ARM_INDICES = np.r_[0:7, 27:34]
HAND_INDICES = np.r_[7:27, 34:54]
ARM_NAMES = tuple(f"{side}_fr3v2_joint{i}" for side in ("left", "right") for i in range(1, 8))


def array(value, shape):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite array {shape}, got {result.shape}")
    result.flags.writeable = False
    return result


def finite(*values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Times and limits must be finite")


@dataclass(frozen=True)
class Feedback:
    positions: np.ndarray
    sampled_mono: float
    valid_until_mono: float
    clock_epoch: int = 0

    def __post_init__(self):
        object.__setattr__(self, "positions", array(self.positions, (54,)))
        finite(self.sampled_mono, self.valid_until_mono)
        if self.valid_until_mono < self.sampled_mono:
            raise ValueError("Feedback was already stale when sampled")


@dataclass(frozen=True)
class Candidate:
    run_id: str
    generation: int
    request_id: int
    observation_id: str
    sent_mono: float
    received_mono: float
    valid_until_mono: float
    actions: np.ndarray
    observed_state: np.ndarray
    clock_epoch: int = 0

    def __post_init__(self):
        if not self.run_id or not self.observation_id or self.generation < 0 or self.request_id < 0:
            raise ValueError("Invalid candidate identity")
        finite(self.sent_mono, self.received_mono, self.valid_until_mono)
        if self.received_mono < self.sent_mono or self.valid_until_mono < self.sent_mono:
            raise ValueError("Invalid candidate times")
        object.__setattr__(self, "actions", array(self.actions, (50, 54)))
        object.__setattr__(self, "observed_state", array(self.observed_state, (54,)))


@dataclass(frozen=True)
class OutputFrame:
    run_id: str
    generation: int
    frame_id: int
    request_id: int
    positions: np.ndarray
    created_mono: float
    valid_until_mono: float

    def __post_init__(self):
        if not self.run_id or min(self.generation, self.frame_id, self.request_id) < 0:
            raise ValueError("Invalid output identity")
        finite(self.created_mono, self.valid_until_mono)
        if self.valid_until_mono <= self.created_mono:
            raise ValueError("Output must have a positive lifetime")
        object.__setattr__(self, "positions", array(self.positions, (54,)))


def arm_payload(frame):
    return {
        "source": "openpi",
        "session_id": f"{frame.run_id}:{frame.generation}",
        "sequence": frame.frame_id,
        "active_sides": ["left", "right"],
        "joint_names": list(ARM_NAMES),
        "positions": frame.positions[ARM_INDICES].tolist(),
    }
