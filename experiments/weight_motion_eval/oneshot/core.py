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

from .limits import ARM_TRACKING_TOLERANCE_RAD
from .limits import ARM_ENDPOINT_TOLERANCE_RAD
from .limits import HAND_ENDPOINT_TOLERANCE_RAD
from .limits import HAND_CONTACT_SECONDS
from .limits import HAND_CONTACT_SETTLE_SECONDS
from .limits import HAND_SPEED_RAD_S
from .limits import SLIDER_POSITION_TOLERANCE_RAD
from .limits import check_hand_control

ARM = np.r_[0:7, 27:34]
HAND = np.r_[7:27, 34:54]
SPEED = np.full(54, HAND_SPEED_RAD_S)
SPEED[ARM] = 0.7
SPEED.flags.writeable = False


def checked_arm_speed(value):
    """Per-session arm limit; legacy callers retain 0.7 rad/s."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1.0:
        raise ValueError("Arm speed must be positive, finite and at most 1.0 rad/s")
    return float(value)


def speed_caps(arm_speed_rad_s):
    result = SPEED.copy()
    result[ARM] = checked_arm_speed(arm_speed_rad_s)
    result.flags.writeable = False
    return result


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
    hand_control: str = "strict"
    hand_targets_reached: bool = True
    hand_soft_limits: tuple = ()
    hand_contacts: tuple = ()
    arm_speed_rad_s: float = 0.7

    def __post_init__(self):
        object.__setattr__(self, "positions", vector(self.positions))
        object.__setattr__(self, "velocities", vector(self.velocities))
        check_hand_control(self.hand_control)
        object.__setattr__(self, "_speed", speed_caps(self.arm_speed_rad_s))
        limits = tuple(tuple(item) for item in self.hand_soft_limits)
        indices = set()
        for item in limits:
            if len(item) != 3:
                raise ValueError("Invalid hand soft limit")
            index, direction, bound = item
            if type(index) is not int or index not in HAND or index in indices or type(direction) is not int or direction not in (-1, 1) or not math.isfinite(bound):
                raise ValueError("Invalid hand soft limit")
            indices.add(index)
        if limits and self.hand_control != "slider":
            raise ValueError("Stall soft limits require slider mode")
        object.__setattr__(self, "hand_soft_limits", limits)
        contacts = tuple(tuple(item) for item in self.hand_contacts)
        indices = set()
        for item in contacts:
            if len(item) != 4:
                raise ValueError("Invalid hand contact")
            index, direction, bound, started = item
            if (type(index) is not int or index not in HAND or index in indices
                    or type(direction) is not int or direction not in (-1, 1)
                    or not math.isfinite(bound) or not math.isfinite(started)):
                raise ValueError("Invalid hand contact")
            indices.add(index)
        if contacts and (self.hand_control != "slider" or limits):
            raise ValueError("Timed hand contacts require slider mode without legacy soft limits")
        object.__setattr__(self, "hand_contacts", contacts)
        if len(self.source_times) != 4 or len(self.receipt_times) != 4:
            raise ValueError("Need timestamps for all four device groups")
        object.__setattr__(self, "source_times", tuple(self.source_times))
        object.__setattr__(self, "receipt_times", tuple(self.receipt_times))
        finite(*self.source_times, *self.receipt_times)

    def effective_target(self, target):
        if not self.hand_soft_limits:
            return target
        result = np.asarray(target, dtype=float).copy()
        for index, direction, bound in self.hand_soft_limits:
            if direction * (result[index] - bound) > 0:
                result[index] = bound
        return result

    def contact_indices(self, target):
        # Only the diagnosed loading direction can count as contact. An opening
        # target must still be physically reached; a latched warning is not enough.
        return tuple(index for index, direction, bound, _ in self.hand_contacts
                     if direction * (target[index] - bound) >= 0
                     and direction * (target[index] - self.positions[index]) > 0.005)

    def hands_settled(self, target, tolerance):
        errors = np.abs(self.positions - self.effective_target(target))
        for index in self.contact_indices(target):
            if abs(self.velocities[index]) <= 0.02:
                errors[index] = 0.0
        return bool(np.all(errors[HAND] <= tolerance))

    def check(self, now, epoch):
        finite(now)
        if any(not 0 <= now - started < HAND_CONTACT_SECONDS for _, _, _, started in self.hand_contacts):
            raise ValueError("Hand contact duration exceeded or timestamp is in the future")
        if self.epoch != epoch or not self.healthy:
            raise ValueError("Feedback clock epoch changed or device unhealthy")
        if any(not 0 <= now - t <= 0.15 for t in (*self.source_times, *self.receipt_times)):
            raise ValueError("Missing, future or stale device feedback")
        indices = ARM if self.hand_control == "slider" else np.arange(54)
        exceeded = indices[np.abs(self.velocities[indices]) > self._speed[indices] + 1e-9]
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
                    f"limit={self._speed[index]:.9f} rad/s"
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
        if not self.sent_time <= self.received_time <= self.observation_time + 1.0:
            raise ValueError("Inference response exceeded original 1 s admission deadline")


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


def check_tracking(target, feedback, indices, label):
    errors = np.abs(target[indices] - feedback.positions[indices])
    limits = np.where(np.isin(indices, ARM), ARM_TRACKING_TOLERANCE_RAD, 0.05)
    bad = np.flatnonzero(errors > limits)
    if bad.size:
        j = bad[np.argmax(errors[bad])]
        index = int(indices[j])
        group, offset = next((name, start) for name, start, end in (
            ("left_arm", 0, 7), ("left_hand", 7, 27),
            ("right_arm", 27, 34), ("right_hand", 34, 54)) if start <= index < end)
        raise ValueError(f"{label} error exceeds {limits[j]:.2f} rad: {group}[{index-offset}], "
                         f"target={target[index]:.6f}, measured={feedback.positions[index]:.6f}, "
                         f"error={errors[j]:.6f} rad")


class ConsumerGuard:
    """Must run at the final send boundary, not just when a frame is queued.

    Commit only after submission succeeds; partial device submission is a fault.
    A guard may arm once and may not be reused after stop/failure.
    """

    def __init__(self, lower, upper, *, hand_control="strict", arm_speed_rad_s=0.7):
        self.hand_control = check_hand_control(hand_control)
        self.checked = ARM if hand_control == "slider" else np.arange(54)
        self.arm_speed_rad_s = checked_arm_speed(arm_speed_rad_s)
        self.speed = speed_caps(self.arm_speed_rad_s)
        self.lower, self.upper = vector(lower), vector(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid position bounds")
        self.state = "new"
        self.last = None

    def arm(self, run_id, digest, feedback, now):
        if self.state != "new":
            raise ValueError("Output consumer is single-use")
        feedback.check(now, feedback.epoch)
        if feedback.arm_speed_rad_s != self.arm_speed_rad_s:
            raise ValueError("Arm speed mode mismatch")
        if feedback.hand_control != self.hand_control:
            raise ValueError("Hand control mode mismatch")
        if not run_id or not digest or np.any(np.abs(feedback.velocities[self.checked]) > 0.02):
            raise ValueError("Acquisition requires identity and stationary feedback")
        self.run_id, self.digest, self.epoch = run_id, digest, feedback.epoch
        self.initial = feedback.positions.copy()
        self.state = "armed"

    def validate(self, frame, feedback, now):
        if self.state != "armed":
            raise ValueError("Output consumer is not armed")
        feedback.check(now, self.epoch)
        if feedback.arm_speed_rad_s != self.arm_speed_rad_s:
            raise ValueError("Arm speed mode changed")
        if feedback.hand_control != self.hand_control:
            raise ValueError("Hand control mode changed")
        if (frame.run_id, frame.plan_hash) != (self.run_id, self.digest):
            raise ValueError("Foreign or superseded frame")
        if not frame.created <= now < frame.valid_until:
            raise ValueError("Expired or future command at final consumer")
        if np.any(frame.positions < self.lower) or np.any(frame.positions > self.upper):
            raise ValueError("Command position exceeds joint limits")
        if np.any(np.abs(frame.velocities[self.checked]) > self.speed[self.checked] + 1e-9):
            raise ValueError("Command derivative exceeds joint speed ceiling")
        check_tracking(frame.positions, feedback, self.checked, "Tracking")
        if self.last is None:
            if (
                frame.sequence != 0
                or np.max(np.abs(frame.positions[self.checked] - self.initial[self.checked])) > 0.005
            ):
                raise ValueError("First output must be the checked starting pose")
        else:
            if frame.sequence != self.last.sequence + 1:
                raise ValueError("Duplicate, out-of-order or skipped command")
            dt = frame.created - self.last.created
            if not 0 < dt <= 0.030000001:
                raise ValueError("Consumer command stream stalled; no catch-up")
            if np.any(
                np.abs(frame.positions[self.checked] - self.last.positions[self.checked])
                > self.speed[self.checked] * dt + 1e-8
            ):
                raise ValueError(f"Command slew exceeds arm {self.arm_speed_rad_s:g} rad/s or hand 45 deg/s")

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
        self.hand_control = check_hand_control(self.plan.config.get("hand_control", "strict"))
        self.checked = ARM if self.hand_control == "slider" else np.arange(54)
        self.admission.check()
        finite(self.prepared_at)
        if self.prepared_at < self.admission.received_time:
            raise ValueError("Plan preparation predates inference")
        if self.plan.raw.shape != (self.plan.config.get("execution_steps", 50), 54):
            raise ValueError("This player executes the configured prediction prefix")
        for phase in (self.plan.approach, self.plan.playback):
            speed = np.maximum(np.abs(phase.lower[1]), np.abs(phase.upper[1])) / phase.scale
            if np.any(speed[self.checked] > SPEED[self.checked] + 1e-8):
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
        if feedback.hand_control != self.hand_control:
            raise ValueError("Player and device hand control mismatch")
        if not self.prepared_at <= now <= self.prepared_at + 30:
            raise ValueError("Reviewed one-shot plan must start within 30 seconds of preparation")
        if np.max(np.abs(feedback.positions[self.checked] - self.plan.start[self.checked])) > (ARM_ENDPOINT_TOLERANCE_RAD if self.plan.config.get("continuation") else 0.005):
            raise ValueError("Start pose changed; rebuild from fresh feedback")
        if np.max(np.abs(feedback.velocities[self.checked])) > 0.02:
            raise ValueError("Start requires stationary feedback")

    def start(self, feedback, now):
        self.check_start(feedback, now)
        self.started = now
        # The finite committed plan is distinct from a streaming policy.
        # 1 s still governed admission above, and is never re-stamped.
        self.deadline = now + self.plan.report["total_seconds"] + (2 * HAND_CONTACT_SETTLE_SECONDS if self.hand_control == "slider" else 10)
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
        if feedback.hand_control != self.hand_control:
            raise ValueError("Player and device hand control mismatch")
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
            if feedback.hand_control != self.hand_control:
                raise ValueError("Player and device hand control mismatch")
            if now > self.deadline:
                raise ValueError("One-shot execution exceeded finite lifetime")
            if self.last_tick is not None and not 0 < now - self.last_tick <= 0.030000001:
                raise ValueError("Scheduler clock jump or missed ticks; no catch-up")
            self.last_tick = now
            if self.state in {"approach", "playback"}:
                phase = self.plan.approach if self.state == "approach" else self.plan.playback
                elapsed = min(now - self.phase_started, phase.duration)
                q, dq = phase.sample(elapsed), phase.sample(elapsed, 1)
                if self.hand_control == "slider":
                    index = 0 if self.state == "approach" else min(len(self.plan.raw) - 1, int(elapsed / phase.duration * (len(self.plan.raw) - 1)))
                    q[HAND], dq[HAND] = self.plan.raw[index, HAND], 0.0
                if elapsed >= phase.duration:
                    self.transition(
                        ("playback" if self.plan.config.get("continuation") else "settle_start")
                        if self.state == "approach" else "settle_end", now, "endpoint"
                    )
            else:
                q = self.plan.raw[0] if self.state == "settle_start" else self.plan.raw[-1]
                dq = np.zeros(54)
                settled = (
                    np.max(np.abs(feedback.positions[ARM] - q[ARM])) <= ARM_ENDPOINT_TOLERANCE_RAD
                    and np.max(np.abs(feedback.velocities[self.checked])) <= 0.02
                )
                hand_tolerance = (HAND_ENDPOINT_TOLERANCE_RAD if self.state == "settle_end"
                                  else SLIDER_POSITION_TOLERANCE_RAD if self.hand_control == "slider" else HAND_ENDPOINT_TOLERANCE_RAD)
                if self.hand_control == "strict":
                    settled = settled and feedback.hands_settled(q, hand_tolerance)
                if self.hand_control == "slider":
                    settled = (
                        settled
                        and feedback.hand_targets_reached
                        and feedback.hands_settled(q, hand_tolerance)
                    )
                if settled:
                    if self.settled_since is None:
                        self.settled_since = now
                    if now - self.settled_since >= self.plan.config["settle_seconds"]:
                        if self.state == "settle_end":
                            self.transition("complete", now, "final_contact_settle" if feedback.contact_indices(q) else "final_measured_settle")
                            return None
                        self.transition("playback", now, "initial_measured_settle")
                else:
                    self.settled_since = None
                settle_timeout = HAND_CONTACT_SETTLE_SECONDS if feedback.contact_indices(q) else 5
                if now - self.phase_started > settle_timeout:
                    raise ValueError("Measured endpoint settling timed out")
            check_tracking(q, feedback, self.checked, "Measured tracking")
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
