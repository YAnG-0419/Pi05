"""Finite conservative chunk player. Blocking inference never runs on the 100 Hz loop."""

import hashlib
import numpy as np
from .contract import to_device
from .robot.core import ARM, HAND


class Trajectory:
    def __init__(self, actions, start, config, bounds):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (8, 54) or not np.isfinite(actions).all():
            raise ValueError("Expected finite DP (8,54) actions")
        self.raw = actions.copy()
        self.actions = to_device(actions)
        lo, hi = np.asarray(bounds["lower"]), np.asarray(bounds["upper"])
        if np.any(self.actions[:, ARM] < lo[ARM]) or np.any(
            self.actions[:, ARM] > hi[ARM]
        ):
            raise ValueError("Arm prediction exceeds physical joint limits")
        # Same hand projection as reference pi0.5; preserve raw predictions in logs.
        projected = np.clip(self.actions[:, HAND], lo[HAND], hi[HAND])
        self.hand_projection_count = int(
            np.count_nonzero(projected != self.actions[:, HAND])
        )
        self.actions[:, HAND] = projected
        start = np.asarray(start, dtype=np.float64)
        if (
            start.shape != (54,)
            or not np.isfinite(start).all()
            or np.any(start < lo)
            or np.any(start > hi)
        ):
            raise ValueError("Invalid measured start")
        if (
            np.max(np.abs(self.actions[0, ARM] - start[ARM]))
            > config["max_initial_arm_delta_rad"]
        ):
            raise ValueError("First arm target is too far from current hold")
        if (
            np.max(np.abs(self.actions[0, HAND] - start[HAND]))
            > config["max_initial_hand_delta_rad"]
        ):
            raise ValueError("First hand target is too far from current hold")
        self.knots = np.vstack([start, self.actions])
        delta = np.abs(np.diff(self.knots, axis=0))
        speed = np.full(54, 0.8)
        speed[ARM] = config["target_arm_speed_rad_s"]
        # Quintic rest-to-rest joins: max derivative 1.875, max second derivative 5.774.
        accel = np.full(54, 3.0)
        accel[ARM] = 1.5
        self.durations = np.maximum(
            1 / config["source_hz"],
            np.maximum(
                (1.875 * delta / speed).max(axis=1),
                np.sqrt((5.774 * delta / accel).max(axis=1)),
            ),
        )
        self.durations[0] = max(0.5, self.durations[0])
        self.times = np.r_[0, np.cumsum(self.durations)]
        self.duration = float(self.times[-1])
        self.digest = hashlib.sha256(
            self.knots.tobytes() + self.times.tobytes()
        ).hexdigest()

    def sample(self, t):
        if t <= 0:
            return self.knots[0].copy(), np.zeros(54)
        if t >= self.duration:
            return self.knots[-1].copy(), np.zeros(54)
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        d = self.durations[i]
        u = (t - self.times[i]) / d
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / d
        delta = self.knots[i + 1] - self.knots[i]
        return self.knots[i] + s * delta, ds * delta


def settled(feedback, target):
    return (
        np.max(np.abs(feedback.positions[ARM] - target[ARM])) <= 0.03
        and np.max(np.abs(feedback.velocities[ARM])) <= 0.02
        and feedback.hand_targets_reached
        and feedback.hands_settled(target, 0.25)
    )
