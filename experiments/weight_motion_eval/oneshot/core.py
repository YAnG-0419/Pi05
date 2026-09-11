"""Bounded single-shot lifecycle, separate from the legacy policy freshness contract.

A fresh inference may be admitted once, then planned into a finite trajectory.
Its original timestamps never change. Every generated command carries a separate
short deadline and is checked again by the output consumer. No device imports.
"""

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
import math
import uuid

import numpy as np

from .limits import HAND_SPEED_RAD_S

ARM = np.r_[0:7, 27:34]
HAND = np.r_[7:27, 34:54]
SPEED = np.full(54, HAND_SPEED_RAD_S)
SPEED[ARM] = 0.7
SPEED.flags.writeable = False


def vector(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != (54,) or not np.isfinite(a).all():
        raise ValueError("Expected 54 finite measured/commanded joint values")
    a.flags.writeable = False
    return a


def finite(*values):
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite timestamp or limit")


@dataclass(frozen=True)
class Feedback:
    positions: np.ndarray
    velocities: np.ndarray
    # Four groups, in model order. Both source and receipt ages are checked.
    source_times: tuple[float, ...]
    receipt_times: tuple[float, ...]
    epoch: int = 0
    healthy: bool = True

    def __post_init__(self):
        object.__setattr__(self, "positions", vector(self.positions))
        object.__setattr__(self, "velocities", vector(self.velocities))
        if len(self.source_times) != 4 or len(self.receipt_times) != 4:
            raise ValueError("Need timestamps for all four device groups")
        object.__setattr__(self, "source_times", tuple(self.source_times))
        object.__setattr__(self, "receipt_times", tuple(self.receipt_times))
        finite(*self.source_times, *self.receipt_times)

    def check(self, now, epoch):
        finite(now)
        if self.epoch != epoch or not self.healthy:
            raise ValueError("Feedback clock epoch changed or device unhealthy")
        if any(not 0 <= now - t <= 0.15 for t in (*self.source_times, *self.receipt_times)):
            raise ValueError("Missing, future or stale device feedback")
        exceeded = np.flatnonzero(np.abs(self.velocities) > SPEED + 1e-9)
        if exceeded.size:
            details = []
            for index in exceeded:
                group, offset = next(
                    (name, start)
                    for name, start, end in (
                        ("left_arm", 0, 7),
                        ("left_hand", 7, 27),
                        ("right_arm", 27, 34),
                        ("right_hand", 34, 54),
                    )
                    if start <= index < end
                )
                details.append(
                    f"{group}[{index - offset}] action_index={index} "
                    f"velocity={self.velocities[index]:.9f} rad/s "
                    f"limit={SPEED[index]:.9f} rad/s"
                )
            raise ValueError("Measured joint speed exceeds software ceiling: " + "; ".join(details))


@dataclass(frozen=True)
class Admission:
    observation_time: float
    sent_time: float
    received_time: float
    request_id: str
    checkpoint: str
    epoch: int

    def check(self):
        finite(self.observation_time, self.sent_time, self.received_time)
        if not self.request_id or not self.checkpoint:
            raise ValueError("Missing inference provenance")
        if not 0 <= self.sent_time - self.observation_time <= 0.07:
            raise ValueError("Inference observation exceeded 70 ms send budget")
        if not self.sent_time <= self.received_time <= self.observation_time + 0.2:
            raise ValueError("Inference response exceeded original 200 ms admission deadline")


@dataclass(frozen=True)
class Frame:
    run_id: str
    plan_hash: str
    sequence: int
    created: float
    valid_until: float
    plan_elapsed: float
    positions: np.ndarray
    velocities: np.ndarray
    phase: str

    def __post_init__(self):
        object.__setattr__(self, "positions", vector(self.positions))
        object.__setattr__(self, "velocities", vector(self.velocities))
        finite(self.created, self.valid_until, self.plan_elapsed)
        if self.sequence < 0 or not self.run_id or not self.plan_hash:
            raise ValueError("Invalid output frame identity")
        if not 0 < self.valid_until - self.created <= 0.020000001:
            raise ValueError("Frame lifetime must be at most 20 ms")


def plan_digest(plan):
    h = hashlib.sha256()
    for name in ("raw", "knots", "start"):
        h.update(np.asarray(getattr(plan, name), dtype="<f8").tobytes())
    # Bind actual spline coefficients as well as config; changing a cached
    # interpolator must invalidate the reviewed plan.
    for phase in (plan.approach, plan.playback):
        h.update(np.asarray(phase.spline.t, dtype="<f8").tobytes())
        h.update(np.asarray(phase.spline.c, dtype="<f8").tobytes())
        h.update(np.asarray([phase.scale], dtype="<f8").tobytes())
    h.update(json.dumps(plan.config, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


class ConsumerGuard:
    """Must run at the final send boundary, not just when a frame is queued.

    Commit only after submission succeeds; partial device submission is a fault.
    A guard may arm once and may not be reused after stop/failure.
    """

    def __init__(self, lower, upper):
        self.lower, self.upper = vector(lower), vector(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid position bounds")
        self.state = "new"
        self.last = None

    def arm(self, run_id, digest, feedback, now):
        if self.state != "new":
            raise ValueError("Output consumer is single-use")
        feedback.check(now, feedback.epoch)
        if not run_id or not digest or np.any(np.abs(feedback.velocities) > 0.02):
            raise ValueError("Acquisition requires identity and stationary feedback")
        self.run_id, self.digest, self.epoch = run_id, digest, feedback.epoch
        self.initial = feedback.positions.copy()
        self.state = "armed"

    def validate(self, frame, feedback, now):
        if self.state != "armed":
            raise ValueError("Output consumer is not armed")
        feedback.check(now, self.epoch)
        if (frame.run_id, frame.plan_hash) != (self.run_id, self.digest):
            raise ValueError("Foreign or superseded frame")
        if not frame.created <= now < frame.valid_until:
            raise ValueError("Expired or future command at final consumer")
        if np.any(frame.positions < self.lower) or np.any(frame.positions > self.upper):
            raise ValueError("Command position exceeds joint limits")
        if np.any(np.abs(frame.velocities) > SPEED + 1e-9):
            raise ValueError("Command derivative exceeds joint speed ceiling")
        if np.max(np.abs(frame.positions - feedback.positions)) > 0.05:
            raise ValueError("Tracking error exceeds 0.05 rad")
        if self.last is None:
            if frame.sequence != 0 or np.max(np.abs(frame.positions - self.initial)) > 0.005:
                raise ValueError("First output must be the checked starting pose")
        else:
            if frame.sequence != self.last.sequence + 1:
                raise ValueError("Duplicate, out-of-order or skipped command")
            dt = frame.created - self.last.created
            if not 0 < dt <= 0.030000001:
                raise ValueError("Consumer command stream stalled; no catch-up")
            if np.any(np.abs(frame.positions - self.last.positions) > SPEED * dt + 1e-8):
                raise ValueError("Command slew exceeds arm 0.7 rad/s or hand 30 deg/s")

    def commit(self, frame):
        self.last = frame

    def stop(self):
        self.state = "stopped"


@dataclass
class OneShot:
    plan: object
    admission: Admission
    prepared_at: float
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "prepared"
    reason: str | None = None
    transitions: list = field(default_factory=list)

    def __post_init__(self):
        self.admission.check()
        finite(self.prepared_at)
        if self.prepared_at < self.admission.received_time:
            raise ValueError("Plan preparation predates inference")
        if self.plan.raw.shape != (50, 54):
            raise ValueError("This player executes one complete 50x54 prediction")
        for phase in (self.plan.approach, self.plan.playback):
            speed = np.maximum(np.abs(phase.lower[1]), np.abs(phase.upper[1])) / phase.scale
            if np.any(speed > SPEED + 1e-8):
                raise ValueError("Planned motion exceeds agreed arm/hand speed ceiling")
        if self.plan.report["total_seconds"] > 60:
            raise ValueError("One-shot plan exceeds 60-second motion budget")
        self.digest = plan_digest(self.plan)
        self.sequence = 0
        self.last_tick = None
        self.settled_since = None
        self.phase_started = None

    def transition(self, state, now, reason):
        self.transitions.append({"from": self.state, "to": state, "at": now, "reason": reason})
        self.state, self.phase_started, self.settled_since = state, now, None

    def check_start(self, feedback, now):
        if self.state != "prepared":
            raise ValueError("A one-shot plan cannot restart or resume")
        self._check_plan()
        feedback.check(now, self.admission.epoch)
        if not self.prepared_at <= now <= self.prepared_at + 30:
            raise ValueError("Reviewed one-shot plan must start within 30 seconds of preparation")
        if np.max(np.abs(feedback.positions - self.plan.start)) > 0.005:
            raise ValueError("Start pose changed; rebuild from fresh feedback")
        if np.max(np.abs(feedback.velocities)) > 0.02:
            raise ValueError("Start requires stationary feedback")

    def start(self, feedback, now):
        self.check_start(feedback, now)
        self.started = now
        # The finite committed plan is distinct from a streaming policy.
        # 200 ms still governed admission above, and is never re-stamped.
        self.deadline = now + self.plan.report["total_seconds"] + 10
        self.transition("approach", now, "single_plan_started")

    def _check_plan(self):
        if plan_digest(self.plan) != self.digest:
            raise ValueError("Plan changed since preparation")

    def request_stop(self, now, reason="operator_stop"):
        if self.state not in {"complete", "stopped", "fault"}:
            self.reason = reason
            self.transition("stopping", now, reason)

    def stopped(self, feedback, now):
        if self.state != "stopping":
            raise ValueError("No stop is pending")
        feedback.check(now, self.admission.epoch)
        if np.max(np.abs(feedback.velocities)) > 0.02:
            self.settled_since = None
            return False
        if self.settled_since is None:
            self.settled_since = now
        if now - self.settled_since >= 0.5:
            self.transition("stopped", now, "measured_stop_confirmed")
            return True
        return False

    def tick(self, feedback, now):
        if self.state in {"prepared", "complete", "stopping", "stopped", "fault"}:
            return None
        try:
            self._check_plan()
            feedback.check(now, self.admission.epoch)
            if now > self.deadline:
                raise ValueError("One-shot execution exceeded finite lifetime")
            if self.last_tick is not None and not 0 < now - self.last_tick <= 0.030000001:
                raise ValueError("Scheduler clock jump or missed ticks; no catch-up")
            self.last_tick = now
            if self.state in {"approach", "playback"}:
                phase = self.plan.approach if self.state == "approach" else self.plan.playback
                elapsed = min(now - self.phase_started, phase.duration)
                q, dq = phase.sample(elapsed), phase.sample(elapsed, 1)
                if elapsed >= phase.duration:
                    self.transition("settle_start" if self.state == "approach" else "settle_end", now, "endpoint")
            else:
                q = self.plan.raw[0] if self.state == "settle_start" else self.plan.raw[-1]
                dq = np.zeros(54)
                settled = np.max(np.abs(feedback.positions - q)) <= 0.01 and np.max(np.abs(feedback.velocities)) <= 0.02
                if settled:
                    if self.settled_since is None:
                        self.settled_since = now
                    if now - self.settled_since >= self.plan.config["settle_seconds"]:
                        if self.state == "settle_end":
                            self.transition("complete", now, "final_measured_settle")
                            return None
                        self.transition("playback", now, "initial_measured_settle")
                else:
                    self.settled_since = None
                if now - self.phase_started > 5:
                    raise ValueError("Measured endpoint settling timed out")
            if np.max(np.abs(q - feedback.positions)) > 0.05:
                raise ValueError("Measured tracking error exceeds 0.05 rad")
            frame = Frame(
                self.run_id,
                self.digest,
                self.sequence,
                now,
                min(now + 0.02, self.deadline),
                now - self.started,
                q,
                dq,
                self.state,
            )
            self.sequence += 1
            return frame
        except ValueError as error:
            self.reason = str(error)
            self.transition("fault", now, self.reason)
            return None
