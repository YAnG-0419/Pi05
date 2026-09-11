"""Virtual-clock rehearsal with a simulated actuator; never imports ROS or SDK."""

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

import numpy as np


class State(StrEnum):
    READY = "ready"
    APPROACH = "approach"
    SETTLE_START = "settle_start"
    PLAYBACK = "playback"
    SETTLE_END = "settle_end"
    COMPLETE = "complete"
    PAUSED = "paused"
    FAULT = "fault"


@dataclass
class Rehearsal:
    plan: object
    state: State = State.READY
    phase_started: float = 0.0
    settled_since: float | None = None
    last_tick: float | None = None
    history: list = field(default_factory=list)
    reason: str | None = None

    def move(self, destination, now, reason):
        self.history.append({"from": self.state.value, "to": destination.value, "time": now, "reason": reason})
        self.state, self.phase_started, self.settled_since = destination, now, None

    def fail(self, now, reason):
        self.reason = reason
        self.move(State.FAULT, now, reason)

    def start(self, now, measured):
        if self.state != State.READY:
            raise ValueError("A rehearsal can only be started once")
        if measured is None or np.asarray(measured).shape != (54,) or not np.isfinite(measured).all():
            self.fail(now, "invalid_start_feedback")
            return
        if np.max(np.abs(measured - self.plan.start)) > 0.01:
            self.fail(now, "start_position_changed")
            return
        self.move(State.APPROACH, now, "explicit_virtual_start")

    def pause(self, now):
        if self.state not in {State.COMPLETE, State.FAULT, State.PAUSED}:
            self.move(State.PAUSED, now, "pause_discards_remaining_plan")

    def tick(self, now, measured, velocity, feedback_stamp):
        if self.state in {State.READY, State.COMPLETE, State.PAUSED, State.FAULT}:
            return None
        if not np.isfinite(now) or (
            self.last_tick is not None and (now < self.last_tick or now - self.last_tick > 0.05)
        ):
            self.fail(now, "scheduler_discontinuity")
            return None
        self.last_tick = now
        if (
            measured is None
            or velocity is None
            or feedback_stamp is None
            or np.asarray(measured).shape != (54,)
            or np.asarray(velocity).shape != (54,)
            or not np.isfinite(measured).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite(feedback_stamp)
            or not 0 <= now - feedback_stamp <= 0.15
        ):
            self.fail(now, "feedback_unavailable")
            return None
        if self.state in {State.APPROACH, State.PLAYBACK}:
            phase = self.plan.approach if self.state == State.APPROACH else self.plan.playback
            elapsed = now - self.phase_started
            target = phase.sample(min(elapsed, phase.duration))
            if elapsed >= phase.duration:
                destination = State.SETTLE_START if self.state == State.APPROACH else State.SETTLE_END
                self.move(destination, now, "reference_endpoint_reached")
        else:
            target = self.plan.raw[0] if self.state == State.SETTLE_START else self.plan.raw[-1]
            if np.max(np.abs(measured - target)) <= 0.01 and np.max(np.abs(velocity)) <= 0.02:
                if self.settled_since is None:
                    self.settled_since = now
                if now - self.settled_since >= self.plan.config["settle_seconds"]:
                    destination = State.PLAYBACK if self.state == State.SETTLE_START else State.COMPLETE
                    self.move(destination, now, "simulated_position_and_velocity_settled")
            else:
                self.settled_since = None
            if now - self.phase_started > self.plan.config["settle_seconds"] + 5:
                self.fail(now, "settling_timeout")
                return None
        # Diagnostic tolerances for the mock plant only; not hardware settings.
        if np.max(np.abs(target - measured)) > 0.1:
            self.fail(now, "simulated_tracking_error")
            return None
        return target


def run_rehearsal(plan, *, inject=None):
    if inject not in {None, "feedback_loss", "pause", "scheduler_delay", "tracking_error"}:
        raise ValueError("Unknown rehearsal injection")
    engine = Rehearsal(plan)
    measured, previous_target = plan.start.copy(), plan.start.copy()
    engine.start(0, measured)
    times, targets, feedback, states = [], [], [], []
    dt, now, injected = 0.01, 0.0, False
    maximum = plan.report["total_seconds"] + 12
    while now <= maximum:
        before = measured.copy()
        measured += (previous_target - measured) * (1 - np.exp(-dt / 0.03))
        velocity = (measured - before) / dt
        stamp = now
        if inject and not injected and now >= 0.4:
            injected = True
            if inject == "feedback_loss":
                stamp = now - 0.2
            elif inject == "pause":
                engine.pause(now)
            elif inject == "scheduler_delay":
                now += 0.1
            elif inject == "tracking_error":
                measured += 0.3
        target = engine.tick(now, measured, velocity, stamp)
        if target is not None:
            previous_target = target
            times.append(now)
            targets.append(target.copy())
            feedback.append(measured.copy())
            states.append(engine.state.value)
        if engine.state in {State.COMPLETE, State.FAULT, State.PAUSED}:
            break
        now += dt
    else:
        engine.fail(now, "virtual_duration_exhausted")
    report = {
        "mode": "simulated_rehearsal",
        "hardware_output": False,
        "state": engine.state.value,
        "failure_reason": engine.reason,
        "injection": inject,
        "injection_applied": injected,
        "virtual_seconds": now,
        "frames": len(times),
        "transitions": engine.history,
        "note": "Synthetic first-order actuator and feedback. This does not validate real tracking, stop or collision.",
    }
    return report, {
        "time": np.array(times),
        "command": np.array(targets),
        "feedback": np.array(feedback),
        "state": np.array(states),
    }
