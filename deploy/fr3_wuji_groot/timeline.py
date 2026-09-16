"""Sixteen-step GR00T timeline reusing the proven Pi05 hardware bounds."""

import hashlib

import numpy as np

from deploy.fr3_wuji_fast.timeline import Chunk
from deploy.fr3_wuji_fast.timeline import check_arm_speed
from deploy.fr3_wuji_fast.timeline import prepare_initial
from deploy.fr3_wuji_fast.timeline import sample_nodes
from deploy.fr3_wuji_fast.timeline import smooth_prefix
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import HAND
from experiments.weight_motion_eval.oneshot.core import vector

from .contract import ACTION_DIM
from .contract import ACTION_HORIZON
from .contract import ACTION_HZ


def checked_chunk(raw, admission, number, limits):
    admission.check()
    if type(number) is not int or number < 1:
        raise ValueError("Invalid inference number")
    raw = np.array(raw, dtype=np.float64, copy=True)
    if raw.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(raw).all():
        raise ValueError("GR00T must return exactly 16 x 54 finite absolute joint positions")
    lower, upper = vector(limits["lower"]), vector(limits["upper"])
    if np.any(lower >= upper):
        raise ValueError("Invalid device joint limits")
    bad = np.argwhere((raw[:, ARM] < lower[ARM]) | (raw[:, ARM] > upper[ARM]))
    if len(bad):
        step, column = bad[0]
        raise ValueError(f"Arm prediction outside joint limits: step={step}, action_index={ARM[column]}")
    actions = raw.copy()
    actions[:, HAND] = np.clip(actions[:, HAND], lower[HAND], upper[HAND])
    projected = int(np.count_nonzero(actions != raw))
    digest = hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest()
    return Chunk(number, admission, raw, actions, digest, projected)


class Timeline:
    """Serial GR00T chunks; hold the last target while a fresh chunk is inferred."""

    def __init__(self, first, *, rounds, guidance_steps=3):
        if type(rounds) is not int or not 1 <= rounds <= 1000:
            raise ValueError("GR00T round count must be between 1 and 1000")
        if type(guidance_steps) is not int or not 2 <= guidance_steps < ACTION_HORIZON:
            raise ValueError("GR00T guidance steps must be between 2 and 15")
        self.first = first
        self.rounds = rounds
        self.guidance_steps = guidance_steps
        self.actions = first.actions.copy()
        self.number = 1
        self.started = None
        self.pending_since = None
        self.last_request_id = first.admission.request_id

    def start(self, now):
        if self.started is not None:
            raise ValueError("A stream cannot restart")
        self.started = now

    def sample(self, now):
        if self.started is None or now < self.started:
            raise ValueError("Stream has not started or clock reversed")
        return sample_nodes(self.actions, now - self.started)

    def exhausted(self, now):
        return now - self.started >= len(self.actions) / ACTION_HZ - 1e-10

    def request_due(self, now):
        return self.pending_since is None and self.number < self.rounds and self.exhausted(now)

    def request(self, now):
        if not self.request_due(now):
            raise ValueError("Unexpected GR00T inference trigger")
        self.pending_since = now
        return {"event": "infer", "number": self.number + 1, "triggered_at": now}

    def install(self, chunk, now, limits):
        if self.pending_since is None or chunk.number != self.number + 1:
            raise ValueError("Unexpected, duplicate or out-of-order GR00T prediction")
        admission = chunk.admission
        admission.check()
        if (
            admission.request_id == self.last_request_id
            or admission.checkpoint != self.first.admission.checkpoint
            or admission.epoch != self.first.admission.epoch
            or admission.sent_time < self.pending_since
            or not admission.received_time <= now <= admission.observation_time + 1.0
        ):
            raise ValueError("Foreign, stale or superseded GR00T prediction")
        anchor, _ = self.sample(now)
        actions = smooth_prefix(chunk.actions, anchor, self.guidance_steps)
        if np.any(actions < limits["lower"]) or np.any(actions > limits["upper"]):
            raise ValueError("GR00T chunk splice exceeds joint limits")
        check_arm_speed(actions)
        self.actions = actions
        self.number = chunk.number
        self.started = now
        self.pending_since = None
        self.last_request_id = admission.request_id
        return {
            "event": "chunk_started",
            "number": chunk.number,
            "at": now,
            "latency_offset_steps": 0,
            "sha256": chunk.sha256,
            "projected_hand_values": chunk.projected_hand_values,
            "executed_nodes": actions.tolist(),
        }

    def check_pending(self, now):
        if self.pending_since is not None and now - self.pending_since > 1.0:
            raise RuntimeError("Fresh GR00T prediction unavailable within its 1 second admission deadline")


__all__ = ["Timeline", "checked_chunk", "prepare_initial"]
