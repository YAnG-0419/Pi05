"""Unified dual-arm/dual-hand ownership; used by the isolated device process."""

import time

import numpy as np

from .core import ARM
from .core import ConsumerGuard
from .core import Feedback
from .core import vector

HAND_SLICES = {"left": slice(7, 27), "right": slice(34, 54)}


class DeviceSession:
    def __init__(self, arms, hands, limits, *, execute=False, qualification=None, clock=time.monotonic):
        self.arms, self.hands, self.limits = arms, hands, limits
        self.execute, self.qualification, self.clock = execute, qualification, clock
        self.guard = ConsumerGuard(limits["lower"], limits["upper"])
        self.state = "readonly"
        self.fault = None
        self.stop_errors = []
        self.last = None
        self.last_activity = clock()
        self.last_hold = 0.0
        self.stop_confirmed = False
        self.last_health_check = 0.0
        self.cancel_requested = lambda: False

    def feedback(self):
        now = self.clock()
        q_arm, dq_arm, arm_source, arm_receive = self.arms.feedback(now)
        q, dq = np.empty(54), np.empty(54)
        q[ARM], dq[ARM] = q_arm, dq_arm
        sources, receipts = [], []
        for index, side in enumerate(("left", "right")):
            position, velocity, stamp = self.hands[side].poll(self.clock(), time.time())
            q[HAND_SLICES[side]], dq[HAND_SLICES[side]] = position, velocity
            sources.extend((arm_source[index], stamp))
            receipts.extend((arm_receive[index], self.hands[side].latest_received))
        feedback = Feedback(q, dq, tuple(sources), tuple(receipts))
        feedback.check(self.clock(), 0)
        return feedback

    def prepare(self, run_id, plan_hash, start, finish_policy):
        if not self.execute or self.qualification is None:
            raise RuntimeError("Device bridge is read-only or lacks commissioning evidence")
        if self.state != "readonly" or self.fault:
            raise RuntimeError("Device session cannot be reused")
        if finish_policy not in {"hold", "disable"}:
            raise ValueError("Explicit normal-finish policy required")
        start = vector(start)
        if np.any(start < self.guard.lower) or np.any(start > self.guard.upper):
            raise ValueError("Starting pose exceeds device position limits")
        self.arms.check_ownership()
        self.check_start(start, self.feedback())
        # Check both identities before enabling either device.
        for side, hand in self.hands.items():
            self.qualification.check_hand(side, hand.identity())
        self.state = "acquiring"
        try:
            for hand in self.hands.values():
                if self.cancel_requested():
                    raise RuntimeError("Device acquisition cancelled by disconnected/stopping client")
                hand.enable(qualification=self.qualification, cancelled=self.cancel_requested)
            feedback = self.feedback()
            self.check_start(start, feedback)
            now = self.clock()
            for side, hand in self.hands.items():
                hand.last = (feedback.positions[HAND_SLICES[side]].copy(), now - 0.01)
            self.guard.arm(run_id, plan_hash, feedback, now)
            self.state, self.finish_policy = "armed", finish_policy
            self.last_activity = now
        except BaseException:
            self.stop()
            raise

    @staticmethod
    def check_start(start, feedback):
        if np.max(np.abs(start - feedback.positions)) > 0.005 or np.max(np.abs(feedback.velocities)) > 0.02:
            raise ValueError("Device acquisition changed the checked pose or is not stationary")

    def submit(self, frame):
        if self.state != "armed":
            raise RuntimeError("Session does not accept motion")
        try:
            feedback = self.feedback()
            self.guard.validate(frame, feedback, self.clock())
            self.arms.submit(frame)  # Wait for this exact gateway acknowledgement.
            hand_outputs = {}
            for side, hand in self.hands.items():
                section = HAND_SLICES[side]
                hand_outputs[side] = hand.submit(
                    frame.positions[section],
                    created=frame.created,
                    valid_until=frame.valid_until,
                    now=self.clock(),
                    lower=np.asarray(self.limits["lower"])[section],
                    upper=np.asarray(self.limits["upper"])[section],
                )
            if self.clock() >= frame.valid_until:
                raise ValueError("Device submission exceeded frame deadline")
            self.guard.commit(frame)
            self.last, self.last_activity = frame, self.clock()
            return {"sequence": frame.sequence, "hands": hand_outputs}
        except BaseException as error:
            self.fault = str(error)
            self.stop()
            raise

    def service(self):
        """Runs even without IPC traffic: expired sender or hand feedback stops."""
        now = self.clock()
        telemetry_errors = getattr(self.arms, "hand_feedback_errors", {})
        if self.state in {"armed", "holding"} and telemetry_errors:
            self.fault = "Hand telemetry error: " + str(telemetry_errors)
            self.stop()
            return
        if self.state == "armed":
            deadline = self.last.valid_until if self.last is not None else self.last_activity + 0.03
            if now >= deadline:
                self.fault = "Device process watchdog: motion sender missed its deadline"
                self.stop()
                return
            if now - self.last_health_check >= 0.01:
                try:
                    self.feedback()
                    self.last_health_check = now
                except (ValueError, RuntimeError) as error:
                    self.fault = str(error)
                    self.stop()
        elif self.state == "holding":
            if now - self.last_activity > 0.3:
                self.fault = "Hold monitor disconnected or stalled"
                self.stop()
            elif now - self.last_hold >= 0.01:
                try:
                    self.feedback()
                    for side, hand in self.hands.items():
                        section = HAND_SLICES[side]
                        hand.submit(
                            self.last.positions[section],
                            created=now,
                            valid_until=now + 0.02,
                            now=self.clock(),
                            lower=np.asarray(self.limits["lower"])[section],
                            upper=np.asarray(self.limits["upper"])[section],
                        )
                    self.last_hold = now
                except (ValueError, RuntimeError) as error:
                    self.fault = str(error)
                    self.stop()

    def finish(self):
        if self.state != "armed" or self.last is None or self.last.phase != "settle_end":
            raise RuntimeError("Cannot finish before final measured settling")
        feedback = self.feedback()
        if (
            np.max(np.abs(feedback.velocities)) > 0.02
            or np.max(np.abs(feedback.positions - self.last.positions)) > 0.01
        ):
            raise ValueError("Devices have not settled at the last target")
        self.guard.stop()
        self.arms.stop()  # Last arm frame expires at the unchanged final deadline.
        self.state = "holding"
        self.last_activity = self.clock()
        self.last_hold = self.last.created
        if self.finish_policy == "disable":
            for hand in self.hands.values():
                hand.disable_after_stop(self.clock())
            self.state = "released"
        return {"state": self.state, "physical_stop_confirmed": False}

    def stop(self):
        if self.state in {"stopped", "released"}:
            return {"state": self.state, "physical_stop_confirmed": self.stop_confirmed, "errors": self.stop_errors}
        self.guard.stop()
        self.state = "stopped"
        for device in (self.arms, *self.hands.values()):
            try:
                (device.stop if device is self.arms else device.emergency_stop)()
            except Exception as error:
                self.stop_errors.append(str(error))
        return {"state": self.state, "physical_stop_confirmed": False, "errors": list(self.stop_errors)}

    def confirm_stop(self, timeout=2):
        end, settled = self.clock() + timeout, None
        while self.clock() < end:
            try:
                feedback = self.feedback()
                if np.max(np.abs(feedback.velocities)) <= 0.02:
                    settled = self.clock() if settled is None else settled
                    if self.clock() - settled >= 0.5:
                        self.stop_confirmed = True
                        return True
                else:
                    settled = None
            except (ValueError, RuntimeError):
                settled = None
            time.sleep(0.005)
        return False

    def close(self):
        if any(hand.owns_enable for hand in self.hands.values()):
            self.stop()
            if self.confirm_stop():
                for hand in self.hands.values():
                    try:
                        hand.disable_after_stop(self.clock())
                    except Exception as error:
                        self.stop_errors.append(str(error))
        for hand in self.hands.values():
            if not hand.owns_enable:
                hand.close_readonly()
        # An unconfirmed hand remains owned until process exit; report the
        # uncertainty. No disconnect or returned SDK RPC is called a stop.
