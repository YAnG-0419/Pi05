"""Unified dual-arm/dual-hand ownership; used by the isolated device process."""

from dataclasses import replace
import time

import numpy as np

from .core import ARM
from .core import HAND
from .core import ConsumerGuard
from .core import Feedback
from .core import vector
from .hand import stabilize_hands
from .hand import wait_initial_feedback
from .limits import HAND_ENDPOINT_TOLERANCE_RAD
from .limits import ARM_ENDPOINT_TOLERANCE_RAD
from .limits import check_hand_control
from .motion_gc import MotionGC

HAND_SLICES = {"left": slice(7, 27), "right": slice(34, 54)}


class DeviceSession:
    def __init__(
        self, arms, hands, limits, *, execute=False, qualification=None, clock=time.monotonic, hand_control="strict", continuous=False, defer_motion_gc=False, arm_speed_rad_s=0.7
    ):
        self.motion_gc = MotionGC() if defer_motion_gc else None
        self.continuous = continuous
        self.hand_control = check_hand_control(hand_control)
        self.checked = ARM if hand_control == "slider" else np.arange(54)
        self.arms, self.hands, self.limits = arms, hands, limits
        self.execute, self.qualification, self.clock = execute, qualification, clock
        self.guard = ConsumerGuard(limits["lower"], limits["upper"], hand_control=hand_control,
                                   arm_speed_rad_s=arm_speed_rad_s)
        self.state = "readonly"
        self.fault = None
        self.plan_hashes = set()
        self.stop_errors = []
        self.last = None
        self.last_activity = clock()
        self.last_hold = 0.0
        self.stop_confirmed = False
        self.last_health_check = 0.0
        self.cancel_requested = lambda: False
        self.feedback_timings = {}

    def feedback(self):
        now = self.clock()
        self.feedback_timings = {}
        began_cpu = time.thread_time()
        try:
            q_arm, dq_arm, arm_source, arm_receive = self.arms.feedback(now)
        finally:
            self.feedback_timings["arms_ms"] = (self.clock() - now) * 1000
            self.feedback_timings["arms_cpu_ms"] = (time.thread_time() - began_cpu) * 1000
        q, dq = np.empty(54), np.empty(54)
        q[ARM], dq[ARM] = q_arm, dq_arm
        sources, receipts = [], []
        for index, side in enumerate(("left", "right")):
            began = self.clock()
            began_cpu = time.thread_time()
            try:
                position, velocity, stamp = self.hands[side].poll(began, time.time())
            finally:
                self.feedback_timings[side + "_hand_ms"] = (self.clock() - began) * 1000
                self.feedback_timings[side + "_hand_cpu_ms"] = (time.thread_time() - began_cpu) * 1000
                self.feedback_timings[side + "_hand_parts"] = getattr(self.hands[side], "poll_timings", None)
            q[HAND_SLICES[side]], dq[HAND_SLICES[side]] = position, velocity
            sources.extend((arm_source[index], stamp))
            receipts.extend((arm_receive[index], self.hands[side].latest_received))
        soft_limits = tuple(item for side, hand in self.hands.items()
                            if self.hand_control == "slider" and hasattr(hand, "soft_limit")
                            for item in hand.soft_limit.snapshot(HAND_SLICES[side].start))
        contacts = tuple(item for side, hand in self.hands.items()
                         if self.hand_control == "slider" and hasattr(hand, "contact_grasp")
                         for item in hand.contact_grasp.snapshot(HAND_SLICES[side].start))
        feedback = Feedback(q, dq, tuple(sources), tuple(receipts), hand_control=self.hand_control,
                            hand_soft_limits=soft_limits, hand_contacts=contacts, arm_speed_rad_s=self.guard.arm_speed_rad_s)
        target = feedback.effective_target(self.last.positions) if self.last is not None else None
        reached = self.last is None or all(
            hand.last is not None and np.max(np.abs(hand.last[0] - target[HAND_SLICES[side]])) <= 1e-6
            for side, hand in self.hands.items()
        )
        if not reached:
            feedback = replace(feedback, hand_targets_reached=False)
        feedback.check(self.clock(), 0)
        self.last_health_check = self.clock()
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
        self.refresh_arms()
        self.check_start(start, self.feedback())
        # Check both identities before enabling either device.
        for side, hand in self.hands.items():
            self.qualification.check_hand(side, hand.identity())
        self.state = "acquiring"
        try:
            if self.motion_gc is not None:
                self.motion_gc.begin()
            self.refresh_arms()
            wait_initial_feedback(self.hands, cancelled=self.cancel_requested)
            for hand in self.hands.values():
                if self.cancel_requested():
                    raise RuntimeError("Device acquisition cancelled by disconnected/stopping client")
                hand.enable(qualification=self.qualification, cancelled=self.acquisition_cancelled)
                self.refresh_arms()
            if self.hand_control == "strict":
                stabilize_hands(list(self.hands.values()), cancelled=self.cancel_requested)
            self.refresh_arms()
            feedback = self.feedback()
            self.check_start(start, feedback)
            now = self.clock()
            for side, hand in self.hands.items():
                hand.last = (feedback.positions[HAND_SLICES[side]].copy(), now - 0.01)
            self.guard.arm(run_id, plan_hash, feedback, now)
            self.plan_hashes.add(plan_hash)
            self.state, self.finish_policy = "armed", finish_policy
            self.last_activity = now
        except BaseException:
            self.stop()
            raise

    def acquisition_cancelled(self):
        # Called by SDK enable waiting loops, on the owning ROS thread.
        if hasattr(self.arms, "spin"):
            self.arms.spin(0.0002)
        return self.cancel_requested()

    def refresh_arms(self):
        if hasattr(self.arms, "refresh_feedback"):
            self.arms.refresh_feedback(cancelled=self.cancel_requested)

    def check_start(self, start, feedback):
        if (
            np.max(np.abs(start[self.checked] - feedback.positions[self.checked])) > 0.005
            or np.max(np.abs(feedback.velocities[self.checked])) > 0.02
        ):
            raise ValueError("Device acquisition changed the checked pose or is not stationary")

    def submit(self, frame):
        if self.state != "armed":
            raise RuntimeError("Session does not accept motion")
        timings = {"received_ms": (self.clock() - frame.created) * 1000}
        try:
            feedback = self.feedback()
            timings["feedback_ms"] = (self.clock() - frame.created) * 1000
            timings["feedback_parts"] = self.feedback_timings.copy()
            self.guard.validate(frame, feedback, self.clock())
            timings["checked_ms"] = (self.clock() - frame.created) * 1000
            pipelined = hasattr(self.arms, "begin_submit")
            if pipelined:
                pending = self.arms.begin_submit(frame)
            else:
                self.arms.submit(frame)
            timings["arm_published_ms"] = (self.clock() - frame.created) * 1000
            hand_outputs = {}
            for side, hand in self.hands.items():
                section = HAND_SLICES[side]
                hand_outputs[side] = hand.submit(
                    frame.positions[section],
                    created=frame.created,
                    valid_until=frame.valid_until,
                    now=self.clock(),
                    checked_feedback=(feedback.positions[section], feedback.velocities[section],
                                      feedback.source_times[1 if side == "left" else 3]),
                    lower=self.guard.lower[section],
                    upper=self.guard.upper[section],
                )
                timings[side + "_sent_ms"] = (self.clock() - frame.created) * 1000
            if pipelined:
                self.arms.confirm_submit(frame, pending)
            timings["acknowledged_ms"] = (self.clock() - frame.created) * 1000
            if self.clock() >= frame.valid_until:
                raise ValueError("Device submission exceeded frame deadline")
            self.guard.commit(frame)
            self.last, self.last_activity = frame, self.clock()
            from .ipc import wire_feedback
            warnings = []
            for hand in self.hands.values():
                if hasattr(hand, "fault_status"):
                    warnings.extend(hand.fault_status.drain())
            return {"sequence": frame.sequence, "hands": hand_outputs, "feedback": wire_feedback(feedback),
                    "timings_ms": timings, "warnings": warnings}
        except BaseException as error:
            timings["failed_ms"] = (self.clock() - frame.created) * 1000
            timings["feedback_parts"] = self.feedback_timings.copy()
            self.fault = f"{error}; sequence={frame.sequence}, phase={frame.phase}, timings_ms={timings}"
            self.stop()
            print("设备提交失败: " + self.fault, flush=True)
            raise RuntimeError(self.fault) from error

    def service(self, *, check_feedback=True):
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
            if check_feedback and now - self.last_health_check >= 0.01:
                try:
                    self.feedback()
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
                            lower=self.guard.lower[section],
                            upper=self.guard.upper[section],
                        )
                    self.last_hold = now
                except (ValueError, RuntimeError) as error:
                    self.fault = str(error)
                    self.stop()

    def next_plan(self, run_id, plan_hash, start):
        """Advance a healthy, settled stream without resetting its wire sequence."""
        if not self.continuous or self.state != "armed" or self.fault or self.last is None or self.last.phase != "settle_end":
            raise RuntimeError("Next plan requires a healthy completed chunk")
        if run_id != self.guard.run_id or not plan_hash or plan_hash in self.plan_hashes:
            raise ValueError("Next plan identity is invalid or repeated")
        start = vector(start)
        if np.max(np.abs(start - self.last.positions)) > 1e-9:
            raise ValueError("Next plan must start at the held command")
        self.check_endpoint()
        self.guard.digest = plan_hash
        self.plan_hashes.add(plan_hash)
        return {"sequence": self.last.sequence + 1}

    def check_endpoint(self):
        feedback = self.feedback()
        if (np.max(np.abs(feedback.velocities[self.checked])) > 0.02
                or np.max(np.abs(feedback.positions[ARM] - self.last.positions[ARM])) > ARM_ENDPOINT_TOLERANCE_RAD):
            raise ValueError("Devices have not settled at the last target")
        tolerance = HAND_ENDPOINT_TOLERANCE_RAD
        if not feedback.hand_targets_reached or not feedback.hands_settled(self.last.positions, tolerance):
            raise ValueError("Hands have not settled at the last target")

    def finish(self):
        if self.state != "armed" or self.last is None or self.last.phase != "settle_end":
            raise RuntimeError("Cannot finish before final measured settling")
        feedback = self.feedback()
        if (
            np.max(np.abs(feedback.velocities[self.checked])) > 0.02
            or np.max(np.abs(feedback.positions[ARM] - self.last.positions[ARM])) > ARM_ENDPOINT_TOLERANCE_RAD
        ):
            raise ValueError("Devices have not settled at the last target")
        if self.hand_control == "strict" and (
            np.max(np.abs(feedback.positions[HAND] - feedback.effective_target(self.last.positions)[HAND])) > HAND_ENDPOINT_TOLERANCE_RAD
        ):
            raise ValueError("Hands have not settled at the last target")
        if self.hand_control == "slider" and (
            not feedback.hand_targets_reached
            or not feedback.hands_settled(self.last.positions, HAND_ENDPOINT_TOLERANCE_RAD)
        ):
            raise ValueError("Hands did not reach UI position tolerance")
        self.guard.stop()
        self.arms.stop()  # Last arm frame expires at the unchanged final deadline.
        self.state = "holding"
        self.last_activity = self.clock()
        self.last_hold = self.last.created
        if self.finish_policy == "disable":
            for hand in self.hands.values():
                hand.disable_after_stop(self.clock())
            self.state = "released"
            if self.motion_gc is not None:
                self.motion_gc.end()
        return {"state": self.state, "physical_stop_confirmed": False}

    def stop(self):
        if self.state in {"stopped", "released"}:
            return {"state": self.state, "physical_stop_confirmed": self.stop_confirmed, "errors": self.stop_errors}
        self.guard.stop()
        self.state = "stopped"
        for hand in self.hands.values():
            if getattr(hand, "trace", None) is not None:
                hand.trace.trigger(self.fault or "stop requested")
        for device in (self.arms, *self.hands.values()):
            try:
                (device.stop if device is self.arms else device.emergency_stop)()
            except Exception as error:
                self.stop_errors.append(str(error))
        if self.motion_gc is not None:
            self.motion_gc.end()
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
