"""Deterministic single-trial scheduler. No model, ROS, SDK, threads or disk I/O."""

from dataclasses import dataclass

import numpy as np

from executor.protocol import ARM_INDICES
from executor.protocol import GROUPS
from executor.protocol import HAND_INDICES
from executor.protocol import OutputFrame
from executor.protocol import array
from executor.protocol import finite
from executor.state_machine import State
from executor.state_machine import StateMachine


@dataclass(frozen=True)
class Limits:
    lower: np.ndarray
    upper: np.ndarray
    speed: np.ndarray
    arm_initial_delta: float = 0.05
    hand_initial_delta: float | None = None
    tick_seconds: float = 0.01
    max_tick_lateness: float = 0.02  # Shadow diagnostic setting, not hardware qualification.

    def __post_init__(self):
        for name in ("lower", "upper", "speed"):
            object.__setattr__(self, name, array(getattr(self, name), (54,)))
        finite(self.arm_initial_delta, self.tick_seconds, self.max_tick_lateness)
        if np.any(self.lower >= self.upper) or np.any(self.speed <= 0):
            raise ValueError("Invalid position/speed bounds")
        if min(self.arm_initial_delta, self.tick_seconds, self.max_tick_lateness) <= 0:
            raise ValueError("Invalid scheduler limits")
        if self.hand_initial_delta is not None:
            finite(self.hand_initial_delta)
            if self.hand_initial_delta <= 0:
                raise ValueError("Hand acquisition threshold must be positive")


class Scheduler:
    def __init__(self, limits, sink, machine=None):
        if getattr(sink, "hardware_output", None) is not False:
            raise ValueError("Only shadow sinks are supported by this scheduler")
        self.limits = limits
        self.machine = machine or StateMachine(run_id=sink.run_id)
        self.sink = sink
        self.pending = None
        self.last_request = -1
        self.frame_id = 0
        self.last_tick = None

    def offer(self, candidate):
        if (candidate.run_id, candidate.generation) != (self.machine.run_id, self.machine.generation):
            return "stale_generation"
        if candidate.request_id <= self.last_request:
            return "duplicate_or_out_of_order_request"
        if self.machine.state not in {State.READY, State.ACQUIRING}:
            return "state_not_ready"
        self.last_request = candidate.request_id
        self.pending = candidate  # Capacity one: newest candidate replaces older uncommitted candidate.
        return None

    def inspect(self, candidate, feedback, now):
        finite(now)
        reasons, metrics = [], {}
        if (candidate.run_id, candidate.generation) != (self.machine.run_id, self.machine.generation):
            reasons.append("stale_generation")
        if self.machine.state not in {State.READY, State.ACQUIRING}:
            reasons.append("state_not_ready")
        if now < candidate.received_mono:
            reasons.append("future_candidate")
        if now >= candidate.valid_until_mono:
            reasons.append("observation_expired")
        if feedback is None:
            reasons.append("feedback_missing")
        else:
            if feedback.clock_epoch != candidate.clock_epoch:
                reasons.append("clock_epoch_changed")
            if not feedback.sampled_mono <= now < feedback.valid_until_mono:
                reasons.append("feedback_expired_or_future")
            if np.any((feedback.positions < self.limits.lower) | (feedback.positions > self.limits.upper)):
                reasons.append("measured_state_out_of_range")
        for group, section in GROUPS.items():
            actions = candidate.actions[:, section]
            if np.any((actions < self.limits.lower[section]) | (actions > self.limits.upper[section])):
                reasons.append("range:" + group)
            metrics[group] = {
                "nominal_speed_max_rad_s": float(np.max(np.abs(np.diff(actions, axis=0))) * 30),
                "nominal_speed_exceedances": int(
                    (np.abs(np.diff(actions, axis=0)) * 30 > self.limits.speed[section]).sum()
                ),
            }
            if feedback is not None:
                delta = float(np.max(np.abs(actions[0] - feedback.positions[section])))
                metrics[group]["first_delta_rad"] = delta
                threshold = self.limits.arm_initial_delta if group.endswith("arm") else self.limits.hand_initial_delta
                if threshold is not None and delta > threshold:
                    reasons.append("initial_delta:" + group)
        if self.limits.hand_initial_delta is None:
            reasons.append("hand_acquisition_unconfigured")
        return {
            "request_id": candidate.request_id,
            "observation_id": candidate.observation_id,
            "generation": candidate.generation,
            "accepted": not reasons,
            "reasons": reasons,
            "metrics": metrics,
            "remaining_ms": (candidate.valid_until_mono - now) * 1000,
            "hardware_output": False,
            "evaluated_mono": now,
        }

    def pause(self, now, reason="pause"):
        self.pending = None
        self.machine.pause(now, reason)
        self.sink.fence(self.machine.run_id, self.machine.generation)
        if self.machine.state == State.STOPPING and self.sink.stop():
            self.machine.stopped(now)

    def fault(self, reason, now):
        self.pending = None
        self.machine.fault(reason, now)
        self.sink.fence(self.machine.run_id, self.machine.generation)
        self.sink.stop()

    def resume(self, now):
        self.machine.resume(now)
        self.pending = None
        self.sink.fence(self.machine.run_id, self.machine.generation)

    def tick(self, now, feedback):
        finite(now)
        previous = self.last_tick
        self.last_tick = now
        if previous is not None and now < previous:
            self.fault("clock_reversed", now)
            return None
        if self.machine.state not in {State.ACQUIRING, State.EXECUTING}:
            return None
        if previous is not None and now - previous > self.limits.tick_seconds + self.limits.max_tick_lateness:
            self.fault("scheduler_overrun", now)
            return None
        if feedback is None or not feedback.sampled_mono <= now < feedback.valid_until_mono:
            self.fault("feedback_unavailable", now)
            return None
        candidate = self.pending
        if candidate is None:
            self.pause(now, "queue_empty")
            return None
        decision = self.inspect(candidate, feedback, now)
        self.pending = None
        if not decision["accepted"]:
            if self.machine.state == State.ACQUIRING:
                self.machine.reject(now)
            else:
                self.fault("candidate_rejected", now)
            return decision
        self.machine.begin(now)
        # Preview the first bounded target, never replace the raw prediction.
        # Acquisition has ALREADY passed. This cannot bypass an initial jump.
        raw = candidate.actions[0]
        step = np.clip(
            raw - feedback.positions,
            -self.limits.speed * self.limits.tick_seconds,
            self.limits.speed * self.limits.tick_seconds,
        )
        frame = OutputFrame(
            self.machine.run_id,
            self.machine.generation,
            self.frame_id,
            candidate.request_id,
            feedback.positions + step,
            now,
            min(candidate.valid_until_mono, feedback.valid_until_mono),
        )
        self.frame_id += 1
        try:
            self.sink.submit(frame, now)
        except Exception:
            self.fault("sink_submission_failed", now)
            raise
        decision.update(
            frame_id=frame.frame_id,
            raw_first=raw.tolist(),
            shadow_target=frame.positions.tolist(),
            arm_payload=frame.positions[ARM_INDICES].tolist(),
            hand_payload=frame.positions[HAND_INDICES].tolist(),
        )
        self.pause(now, "single_shadow_step_complete")
        return decision
