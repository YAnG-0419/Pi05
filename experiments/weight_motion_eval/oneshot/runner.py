"""Device-independent runner. Device adapters must check deadlines at submission."""

import time

import numpy as np

from .core import Admission
from .core import ConsumerGuard
from .core import Feedback
from .core import OneShot


class SimulatedDevices:
    hardware_output = False

    def __init__(self, start, lower, upper, *, inject=None):
        self.guard = ConsumerGuard(lower, upper)
        self.positions = np.array(start, copy=True)
        self.target = self.positions.copy()
        self.velocity = np.zeros(54)
        self.now = 0.0
        self.inject = inject
        self.submitted = []
        self.stop_requests = 0

    def feedback(self, now):
        dt = now - self.now
        if dt > 0:
            before = self.positions.copy()
            self.positions += (self.target - self.positions) * (1 - np.exp(-dt / 0.03))
            self.velocity = (self.positions - before) / dt
            self.now = now
        stamp = now - 0.16 if self.inject == "feedback_loss" and now > 0.5 else now
        return Feedback(self.positions, self.velocity, (stamp,) * 4, (stamp,) * 4)

    def prepare(self, player, feedback, now):
        self.guard.arm(player.run_id, player.digest, feedback, now)

    def submit(self, frame, now):
        if self.inject == "consumer_delay" and now > 0.5:
            now = frame.valid_until + 0.001
        self.guard.validate(frame, self.feedback(now), now)
        self.target = frame.positions.copy()
        self.submitted.append(frame)
        self.guard.commit(frame)

    def stop(self):
        self.stop_requests += 1
        self.guard.stop()
        self.target = self.positions.copy()


def simulate(plan, limits, *, inject=None):
    if inject not in {None, "feedback_loss", "consumer_delay", "scheduler_delay", "pause", "partial_submit"}:
        raise ValueError("Unknown fault injection")
    # Virtual admission only. No loaded record is represented as fresh hardware feedback.
    admission = Admission(0.0, 0.05, 0.15, "virtual-request", "simulated", 0)
    player = OneShot(plan, admission, 0.2)
    device = SimulatedDevices(plan.start, limits["lower"], limits["upper"], inject=inject)
    now = 0.2
    feedback = device.feedback(now)
    device.prepare(player, feedback, now)
    player.start(feedback, now)
    history, failed = [], None
    while now < player.deadline + 1:
        if inject == "scheduler_delay" and 0.5 < now < 0.52:
            now += 0.05
        if inject == "pause" and now > 0.5:
            player.request_stop(now)
        feedback = device.feedback(now)
        frame = player.tick(feedback, now)
        if frame is not None:
            try:
                device.submit(frame, now)
                if inject == "partial_submit" and now > 0.5:
                    raise RuntimeError("Simulated hand failure after arm submission")
            except (ValueError, RuntimeError) as error:
                failed = str(error)
                player.reason = failed
                player.transition("fault", now, failed)
        history.append((now, feedback.positions.copy(), player.state))
        if player.state in {"complete", "fault", "stopping"}:
            device.stop()
            break
        now += 0.01
    else:
        device.stop()
        raise RuntimeError("Simulation exceeded deadline")
    if player.state == "stopping":
        # Show that a software stop request alone is not a measured stop.
        while now < player.deadline:
            now += 0.01
            if player.stopped(device.feedback(now), now):
                break
    return {
        "mode": "single_shot_simulation",
        "hardware_output": False,
        "state": player.state,
        "reason": player.reason,
        "frames": len(device.submitted),
        "stop_requests": device.stop_requests,
        "transitions": player.transitions,
        "run_id": player.run_id,
        "plan_hash": player.digest,
        "virtual_seconds": now - player.started,
        "injection": inject,
        "note": "Mock actuator only; completion is not evidence of physical stopping.",
    }, {
        "time": np.array([x[0] for x in history]),
        "feedback": np.array([x[1] for x in history]),
        "state": np.array([x[2] for x in history]),
        "command_time": np.array([x.created for x in device.submitted]),
        "command": np.array([x.positions for x in device.submitted]),
    }


def run_live(player, devices, *, stop_requested, emit, clock=time.monotonic, sleep=time.sleep):
    """One prepared plan, bounded timing; caller supplies admitted devices.

    No model or disk work belongs here. `emit` must be a nonblocking bounded
    recorder and must fail when critical events cannot be recorded.
    Device.prepare checks the selected admission mode and live device state.
    """
    now = clock()
    feedback = devices.feedback(now)
    now = clock()  # IPC feedback can be newer than the request's timestamp.
    try:
        # Validate provenance, pose and immutable plan before device ownership.
        player.check_start(feedback, now)
        devices.prepare(player, feedback, now)
        # Acquisition may take seconds. Start the trajectory clock only after
        # enabling completes and a second measured-pose check succeeds.
        feedback = devices.feedback(clock())
        player.start(feedback, clock())
        due = clock()
        while player.state not in {"complete", "fault", "stopping"}:
            if stop_requested():
                player.request_stop(clock())
                break
            now = clock()
            if now < due:
                sleep(due - now)
            now = clock()
            if now - due > 0.01:
                raise RuntimeError("Output deadline missed; no burst catch-up")
            feedback = devices.feedback(now)
            now = clock()
            if now - due > 0.01:
                raise RuntimeError("Feedback acquisition missed the output deadline")
            frame = player.tick(feedback, now)
            if frame is not None:
                devices.submit(frame, clock())
                emit(frame, feedback)
            due += 0.01
            if player.state not in {"complete", "fault", "stopping"} and clock() > due:
                raise RuntimeError("Submission overran the next 100 Hz tick; no catch-up")
        if player.state == "complete":
            devices.finish()  # Must preserve the selected end-state policy.
        else:
            devices.stop()  # Must not claim a physical stop based on send() returning.
    except BaseException:
        devices.stop()
        raise
    return player
