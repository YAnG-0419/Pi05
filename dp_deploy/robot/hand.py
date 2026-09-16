"""Explicit SDK boundary; connection/readback never enables or disables a hand.

The owner runs in the existing SDK environment. It must be supervised externally;
Python checks alone cannot establish device behaviour after process/network loss.
"""

import time

import numpy as np

from .hand_contact import BoundedContactGrasp
from .hand_faults import HandFaults
from .hand_trace import HandTrace
from .limits import HAND_CLOCK_LEAD_SECONDS
from .limits import HAND_CONTACT_SECONDS
from .limits import HAND_CURRENT_A
from .limits import HAND_KD
from .limits import HAND_KP
from .limits import HAND_SPEED_RAD_S
from .limits import SLIDER_CURRENT_A
from .limits import SLIDER_SPEED_RAD_S
from .limits import check_hand_control
from .poll_timing import PollTiming

NIDS = tuple(finger * 5 + joint + 1 for finger in range(5) for joint in range(4))
NID_SET = frozenset(NIDS)


class HandFeedbackUnavailable(ValueError):  # noqa: N818 - existing public exception name
    """Missing or stale feedback that may recover before startup admission."""


def check_feedback_age(age, label):
    detail = f"{label}: age={age:.9f}s, allowed=[{-HAND_CLOCK_LEAD_SECONDS:.6f}, 0.150000]s"
    if age < -HAND_CLOCK_LEAD_SECONDS:
        raise ValueError("Hand feedback is stale or from a future clock; " + detail)
    if age > 0.15:
        raise HandFeedbackUnavailable("Hand feedback is stale or from a future clock; " + detail)


def wait_initial_feedback(hands, *, cancelled=lambda: False, clock=time.monotonic,
                          wall_clock=time.time, sleep=time.sleep, timeout=3.0, report=print):
    """Drain connection backlog while disabled; never retry firmware/clock faults."""
    deadline = clock() + timeout
    pending = dict(hands)
    errors = {}
    while pending:
        if cancelled():
            raise RuntimeError("Initial hand feedback cancelled")
        for side, hand in list(pending.items()):
            try:
                hand.poll(clock(), wall_clock())
            except HandFeedbackUnavailable as error:
                detail = str(error)
                if side not in errors:
                    report(f"Waiting for initial {side} hand feedback: {detail}")
                errors[side] = detail
            else:
                report(f"Initial {side} hand feedback ready")
                del pending[side]
        if pending:
            if clock() >= deadline:
                raise RuntimeError("Initial hand feedback timed out: " + "; ".join(
                    f"{side}: {errors[side]}" for side in pending))
            sleep(0.005)


def stabilize_hands(hands, *, cancelled=lambda: False, clock=time.monotonic, sleep=time.sleep):
    """Hold all acquired hands together; each requires 0.5 s of fresh quiet data."""
    deadline = clock() + 3.0
    while clock() < deadline:
        if cancelled():
            raise RuntimeError("Hand stabilization cancelled")
        started = clock()
        # Do not short-circuit: every hand must receive its holding command.
        settled = [hand.hold_initial() for hand in hands]
        if all(settled):
            return
        sleep(max(0.0, 0.01 - (clock() - started)))
    raise RuntimeError("Hand stabilization timed out after 3 seconds")


def decode_state(frame, now_wall, now_mono):
    entries = {int(j.nid): j for j in frame.joints}
    if len(frame.joints) != 20 or int(frame.num_joints) != 20 or set(entries) != set(NIDS):
        raise ValueError("Hand state must contain exactly the expected 20 firmware joint IDs")
    values = np.array([[entries[n].position, entries[n].velocity] for n in NIDS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite hand feedback")
    age = now_wall - int(frame.header.timestamp_us) / 1e6
    check_feedback_age(age, "hand state")
    return values[:, 0], values[:, 1], now_mono - max(0.0, age)


class HandOwner:
    def __init__(self, sdk, side, address, *, hand_control="strict"):
        self.hand_control = check_hand_control(hand_control)
        if side not in {"left", "right"} or not address:
            raise ValueError("Explicit hand identity required")
        self.sdk, self.side = sdk, side
        self.hand = sdk.SdkManager.instance().connect(
            address=address,
            device_name="pi05_oneshot_" + side,
            options=sdk.ConnectOptions(enable_bridge=False, auto_time_sync_interval_ms=None),
        )
        self.state_sub = self.diagnostic_sub = self.publisher = None
        self.owns_enable = False
        self.faulted = False
        self.last = None
        self.latest = None
        self.latest_received = None
        self.diagnostics = None
        self.trace = HandTrace()
        self.contact_grasp = BoundedContactGrasp()
        self.fault_status = HandFaults(sdk, side)
        self.trace_identity = None
        self.hold_start = self.hold_quiet_since = self.hold_last_stamp = self.hold_last_sent = None
        try:
            if str(self.hand.handedness().get()).lower() != side or int(self.hand.online_joints_count().get()) != 20:
                raise ValueError("Hand identity/online joint count mismatch")
            self.gains = self.hand.mit_params().get()
            self.effort_limits = self.hand.effort_limit().get()
            if len(self.gains) != 20 or any(x is None for x in self.gains):
                raise ValueError("Incomplete hand MIT parameter readback")
            if len(self.effort_limits) != 20 or any(x is None for x in self.effort_limits):
                raise ValueError("Incomplete hand current-limit readback")
            self.state_sub = self.hand.joint_states().subscribe()
            self.diagnostic_sub = self.hand.joint_diagnostics().subscribe()
            self.trace_identity = {
                "serial": str(getattr(self.hand, "serial_number", "unknown")),
                "info": repr(getattr(self.hand, "info", None)),
                "mit_gains": [{"kp": float(x.kp), "kd": float(x.kd)} for x in self.gains],
                "effort_limits": [float(x) for x in self.effort_limits],
            }
        except BaseException:
            self.close_readonly()
            raise

    def poll(self, now_mono, now_wall):
        timing = PollTiming()
        try:
            with timing:
                return self._poll(now_mono, now_wall, timing)
        finally:
            self.poll_timings = timing.result
            # Keep slow idle/telemetry polls too, not only command submissions.
            # No formatting, console output or disk I/O on this path.
            if timing.result["total_ms"] >= 2 and getattr(self, "trace", None) is not None:
                self.trace.add("slow_feedback_poll", timings=timing.result)

    def _poll(self, now_mono, now_wall, timing):
        # Bound draining work so a producer cannot starve stop processing.
        started = time.monotonic()
        newest_state = newest_diagnostics = None
        for _ in range(256):
            timing.mark("state_recv_ms")
            frame = self.state_sub.recv()
            timing.mark("other_ms")
            if frame is None:
                break
            if getattr(self, "trace", None) is not None:
                timing.mark("trace_ms")
                self.trace.frame(frame)
                timing.mark("other_ms")
            timing.counts["state_frames"] += 1
            newest_state = frame
        for _ in range(64):
            timing.mark("diagnostic_recv_ms")
            frame = self.diagnostic_sub.recv()
            timing.mark("other_ms")
            if frame is None:
                break
            if getattr(self, "trace", None) is not None:
                timing.mark("trace_ms")
                self.trace.frame(frame, diagnostic=True)
            timing.mark("diagnostic_check_ms")
            timing.counts["diagnostic_frames"] += 1
            by_id = {int(j.nid): j for j in frame.joints}
            if len(frame.joints) != 20 or by_id.keys() != NID_SET:
                raise ValueError("Missing or duplicate hand diagnostic joint")
            if not hasattr(self, "fault_status"):
                self.fault_status = HandFaults(getattr(self, "sdk", None), getattr(self, "side", "unknown"))
            self.fault_status.check(frame.joints, started)
            newest_diagnostics = (by_id, int(frame.header.timestamp_us) / 1e6)
        # A frame can arrive while draining. Compare it to the post-read clock,
        # not the earlier timestamp at function entry. Only the newest complete
        # state is decoded, while firmware errors in any drained diagnostic latch.
        timing.mark("decode_ms")
        elapsed = time.monotonic() - started
        now_mono, now_wall = now_mono + elapsed, now_wall + elapsed
        if newest_state is not None:
            self.latest = decode_state(newest_state, now_wall, now_mono)
            self.latest_received = now_mono
        if newest_diagnostics is not None:
            by_id, stamp = newest_diagnostics
            age = now_wall - stamp
            check_feedback_age(age, "hand diagnostics")
            self.diagnostics = (by_id, now_mono - max(0.0, age))
        if self.latest is None or now_mono - self.latest[2] > 0.15:
            raise HandFeedbackUnavailable("No fresh measured hand state")
        if self.diagnostics is None or now_mono - self.diagnostics[1] > 0.15:
            raise HandFeedbackUnavailable("No fresh hand diagnostics")
        if getattr(self, "hand_control", "strict") == "slider" and self.owns_enable and not self.faulted:
            self.contact_grasp.check_time(now_mono)
        return self.latest

    def identity(self):
        self.gains = self.hand.mit_params().get()
        self.effort_limits = self.hand.effort_limit().get()
        return {
            "serial": str(self.hand.serial_number),
            "info": repr(self.hand.info),
            "mit_gains": [{"kp": float(x.kp), "kd": float(x.kd)} for x in self.gains],
            "effort_limits": [float(x) for x in self.effort_limits],
        }

    def configure_deployment(self):
        """Explicit execution-only setup, while disabled; never called by connect."""
        if self.owns_enable or self.faulted:
            raise RuntimeError("Cannot configure an owned or faulted hand")
        _, velocity, _ = self.poll(time.monotonic(), time.time())
        if (getattr(self, "hand_control", "strict") == "strict" and np.max(np.abs(velocity)) > 0.02) or any(
            j.status_word.ext_state != 1 for j in self.diagnostics[0].values()
        ):
            raise RuntimeError("Parameter setup requires all 20 joints Ready and stationary")
        current_limit = SLIDER_CURRENT_A if getattr(self, "hand_control", "strict") == "slider" else HAND_CURRENT_A
        before = self.identity()
        self.trace.add("deployment_parameters_before", identity=before)
        self.trace.add("deployment_parameters_write_attempt", kp=HAND_KP, kd=HAND_KD, effort_a=current_limit)
        # Lower/set the current ceiling before applying the agreed stiffness.
        self.hand.effort_limit().set(current_limit)
        self.hand.mit_params().set([(HAND_KP, HAND_KD)] * 20)
        actual = self.identity()
        self.trace.add("deployment_parameters_readback", identity=actual)
        gains = [[g["kp"], g["kd"]] for g in actual["mit_gains"]]
        if len(gains) != 20 or not np.allclose(gains, [HAND_KP, HAND_KD], rtol=1e-6, atol=1e-7):
            raise RuntimeError("Deployment MIT parameter readback mismatch; hand not enabled")
        if len(actual["effort_limits"]) != 20 or not np.allclose(
            actual["effort_limits"], current_limit, rtol=1e-6, atol=1e-7
        ):
            raise RuntimeError("Deployment current limit readback mismatch; hand not enabled")
        self.trace_identity = actual
        return actual

    def enable(self, *, qualification=None, cancelled=lambda: False):
        from .qualification import Qualification
        from .qualification import SupervisedTrial

        if not isinstance(qualification, Qualification | SupervisedTrial):
            raise RuntimeError("Hand process-loss/network-loss stopping has not been measured")
        qualification.check_hand(self.side, self.identity())
        if cancelled():
            raise RuntimeError("Hand acquisition cancelled before enabling")
        if self.faulted or self.owns_enable:
            raise RuntimeError("Hand owner cannot be re-enabled")
        position, _, _ = self.poll(time.monotonic(), time.time())
        if any(j.status_word.ext_state == 2 for j in self.diagnostics[0].values()):
            raise RuntimeError("Hand is already enabled; ownership must be resolved before acquiring it")
        if any(j.status_word.ext_state != 1 for j in self.diagnostics[0].values()):
            raise RuntimeError("All hand joints must confirm disabled before acquisition")
        self.publisher = self.hand.joint_command().publish()
        # Prime the current measured target while still disabled. Enabling
        # must not reactivate an old target retained by firmware.
        self.publisher.send([self.sdk.JointCommand(float(q), 0.0, 0.0) for q in position])
        self.trace.add(
            "command_sent", phase="enable_prime", positions=position.tolist(), velocities=[0.0] * 20, efforts=[0.0] * 20
        )
        self.owns_enable = True  # Cleanup is required even if enable() raises midway.
        try:
            self.hand.enable()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if cancelled():
                    raise RuntimeError("Hand acquisition cancelled while enabling")
                self.poll(time.monotonic(), time.time())
                if all(j.status_word.ext_state == 2 for j in self.diagnostics[0].values()):
                    self.last = (position.copy(), time.monotonic())
                    self.hold_start = position.copy()
                    self.hold_quiet_since = self.hold_last_stamp = self.hold_last_sent = None
                    self.trace.add("stabilization_begin", positions=position.tolist())
                    return
                time.sleep(0.01)
            raise RuntimeError("All 20 hand joints did not confirm enabled state")
        except BaseException:
            self.emergency_stop()
            raise

    def hold_initial(self):
        """Keep the checked acquisition target; small speed transients reset dwell."""
        if not self.owns_enable or self.faulted or self.hold_start is None:
            raise RuntimeError("Cannot stabilize a hand that is not acquired")
        began = time.monotonic()
        position, velocity, stamp = self.poll(began, time.time())
        if any(j.status_word.ext_state != 2 for j in self.diagnostics[0].values()):
            raise RuntimeError("Hand lost enabled state during stabilization")
        if np.max(np.abs(position - self.hold_start)) > 0.005:
            raise RuntimeError("Hand stabilization exceeded initial pose bound (0.005 rad)")
        if np.max(np.abs(velocity)) > HAND_SPEED_RAD_S:
            index = int(np.argmax(np.abs(velocity)))
            raise RuntimeError(
                f"Hand stabilization overspeed: {self.side}[{index}] velocity={velocity[index]:.9f} rad/s"
            )
        now = time.monotonic()
        if now - began >= 0.02 or (self.hold_last_sent is not None and now - self.hold_last_sent > 0.03):
            raise RuntimeError("Hand stabilization command deadline missed")
        # Constant target has zero commanded velocity. No measured-pose reseed,
        # drift following, increment accumulation or advance toward the test target.
        commands = [self.sdk.JointCommand(float(q), 0.0, 0.0) for q in self.hold_start]
        self.publisher.send(commands)
        self.hold_last_sent = time.monotonic()
        self.last = (self.hold_start.copy(), self.hold_last_sent)
        self.trace.add(
            "command_sent",
            phase="enable_stabilize",
            positions=self.hold_start.tolist(),
            velocities=[0.0] * 20,
            efforts=[0.0] * 20,
        )
        if self.hold_last_sent - began >= 0.02:
            raise RuntimeError("Hand stabilization publisher exceeded deadline")
        fresh = self.hold_last_stamp is None or stamp > self.hold_last_stamp
        if not fresh or np.max(np.abs(velocity)) > 0.02:
            self.hold_quiet_since = None
        elif self.hold_quiet_since is None:
            self.hold_quiet_since = stamp
        self.hold_last_stamp = stamp
        ready = self.hold_quiet_since is not None and stamp - self.hold_quiet_since >= 0.5
        if ready:
            self.trace.add(
                "stabilization_ready",
                position_error_rad=float(np.max(np.abs(position - self.hold_start))),
                max_velocity_rad_s=float(np.max(np.abs(velocity))),
            )
        return ready

    def submit(self, positions, *, created, valid_until, now, lower, upper, checked_feedback=None):
        if not self.owns_enable or self.faulted or self.last is None:
            raise RuntimeError("Hand is not owned/enabled")
        q = np.asarray(positions, dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError("Invalid hand target")
        if not created <= now < valid_until or not 0 < valid_until - created <= 0.020000001:
            raise ValueError("Hand command expired at SDK boundary")
        if np.any(q < lower) or np.any(q > upper):
            raise ValueError("Hand command exceeds position limits")
        if checked_feedback is None:
            measured, velocity, stamp = self.poll(now, time.time())
        else:
            # DeviceSession just drained/validated these streams before gateway
            # submission. Recheck age at send; do not drain the same queues twice.
            measured, velocity, stamp = checked_feedback
            current = time.monotonic()
            if not 0 <= current - stamp <= 0.15 or self.diagnostics is None or not 0 <= current - self.diagnostics[1] <= 0.15:
                raise ValueError("Checked hand feedback expired before submission")

        if getattr(self, "hand_control", "strict") == "strict" and (
            np.any(np.abs(velocity) > HAND_SPEED_RAD_S) or np.max(np.abs(q - measured)) > 0.05
        ):
            raise ValueError("Hand actual speed/tracking error exceeds limits")
        if any(j.status_word.ext_state != 2 for j in self.diagnostics[0].values()):
            raise ValueError("Hand lost enabled state")
        # Limit against the last command actually sent, using elapsed time at
        # this SDK boundary. Do not accumulate spare travel or catch up after a
        # stalled stream. The common planner already retimes the whole path;
        # this final clamp also handles arrival-time jitter between devices.
        send_at = time.monotonic()
        if not created <= send_at < valid_until:
            raise ValueError("Hand command expired during feedback checks")
        dt = send_at - self.last[1]
        if not 0 < dt <= 0.030000001:
            raise ValueError("Hand command stream stalled or clock reversed")
        effective = q
        if getattr(self, "hand_control", "strict") == "slider":
            if not hasattr(self, "contact_grasp"):
                self.contact_grasp = BoundedContactGrasp()
            faults = getattr(self, "fault_status", None)
            stalled = set() if faults is None else {NIDS.index(nid) for nid in faults.stalled_nids()}
            effective, contact_events = self.contact_grasp.update(q, measured, self.last[0], stalled, send_at)
            for event in contact_events:
                nid = NIDS[event["index"]]
                if getattr(self, "trace", None) is not None:
                    self.trace.add("bounded_contact", nid=nid, **event)
                if faults is not None:
                    side = "左手" if getattr(self, "side", "unknown") == "left" else "右手"
                    action = f"开始（电流上限{SLIDER_CURRENT_A:g}A，最长{HAND_CONTACT_SECONDS:g}s）" if event["action"] == "engaged" else "退回后释放"
                    faults.pending.append(f"{side} NID={nid} 接触抓取{action}; "
                                          f"方向={event['direction']}, 触发实测={event['measured_at_trigger']:.5f}rad, "
                                          f"触发指令={event['bound']:.5f}rad")
        delta = effective - self.last[0]
        speed = SLIDER_SPEED_RAD_S if getattr(self, "hand_control", "strict") == "slider" else HAND_SPEED_RAD_S
        allowed = speed * dt
        limited = self.last[0] + np.clip(delta, -allowed, allowed)
        # Do not accidentally use this field as a velocity limiter: MIT velocity
        # is a feedforward target. The limit is enforced on q(t) above.
        if time.monotonic() >= valid_until:
            raise ValueError("Hand command expired during feedback checks")
        commands = [self.sdk.JointCommand(float(x), 0.0, 0.0) for x in limited]
        if getattr(self, "trace", None) is not None:
            self.trace.add(
                "command_attempt",
                requested_positions=q.tolist(),
                positions=limited.tolist(),
                velocities=[0.0] * 20,
                efforts=[0.0] * 20,
                created=created,
                valid_until=valid_until,
            )
        if time.monotonic() >= valid_until:
            raise ValueError("Hand command expired before publisher send")
        self.publisher.send(commands)
        if getattr(self, "trace", None) is not None:
            self.trace.add("command_sent", positions=limited.tolist(), velocities=[0.0] * 20, efforts=[0.0] * 20)
        # Store completion time: next frame's available interval is conservative
        # even when the publisher call itself takes time. Failed sends don't commit.
        self.last = (limited.copy(), time.monotonic())
        return {
            "positions": limited.tolist(),
            "rate_limited_indices": np.flatnonzero(np.abs(delta) > allowed).tolist(),
            "soft_limited_indices": np.flatnonzero(np.abs(q - effective) > 1e-9).tolist(),
            "effective_targets": effective.tolist(),
            "soft_limits": (),
            "contacts": self.contact_grasp.snapshot() if hasattr(self, "contact_grasp") else (),
            "sent_at": self.last[1],
            "interval_seconds": dt,
        }

    def emergency_stop(self):
        self.faulted = True
        if getattr(self, "hand_control", "strict") == "slider":
            self.trace.trigger("slider disable requested")
            self.trace.add("disable_requested", owns_enable=self.owns_enable)
            if self.owns_enable:
                self.hand.disable()
            return  # Keep ownership until feedback confirms Ready.
        if getattr(self, "trace", None) is not None:
            self.trace.trigger("emergency_stop requested")
            self.trace.add("emergency_stop_requested", owns_enable=self.owns_enable)
        if self.owns_enable:
            self.hand.emergency_stop()
        # Keep the connection/feedback alive for confirmation; a returned RPC
        # is not a claim that the hand has physically stopped.

    def capture_raw(self):
        """Post-stop diagnostics only: never validates, sends, enables or clears."""
        for subscription, diagnostic, count in ((self.state_sub, False, 256), (self.diagnostic_sub, True, 64)):
            for _ in range(count):
                frame = subscription.recv()
                if frame is None:
                    break
                self.trace.frame(frame, diagnostic=diagnostic)

    def close_readonly(self):
        if self.owns_enable:
            raise RuntimeError("Owned hand must be stopped and explicitly released before disconnect")
        for resource in (self.publisher, self.state_sub, self.diagnostic_sub):
            if resource is not None:
                resource.close()
        if self.hand is not None:
            self.hand.disconnect()

    def disable_after_stop(self, now):
        _, velocity, _ = self.poll(now, time.time())
        if getattr(self, "hand_control", "strict") == "strict" and np.max(np.abs(velocity)) > 0.02:
            raise RuntimeError("Cannot release hand before measured stop")
        self.hand.disable()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            self.poll(time.monotonic(), time.time())
            if all(j.status_word.ext_state == 1 for j in self.diagnostics[0].values()):
                self.owns_enable = False
                return
            time.sleep(0.005)
        raise RuntimeError("Hand disable did not confirm all joints released")
