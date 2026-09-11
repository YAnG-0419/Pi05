"""Explicit SDK boundary; connection/readback never enables or disables a hand.

The owner runs in the existing SDK environment. It must be supervised externally;
Python checks alone cannot establish device behaviour after process/network loss.
"""

import time

import numpy as np

from .limits import HAND_SPEED_RAD_S

NIDS = tuple(finger * 5 + joint + 1 for finger in range(5) for joint in range(4))


def decode_state(frame, now_wall, now_mono):
    entries = {int(j.nid): j for j in frame.joints}
    if len(frame.joints) != 20 or int(frame.num_joints) != 20 or set(entries) != set(NIDS):
        raise ValueError("Hand state must contain exactly the expected 20 firmware joint IDs")
    values = np.array([[entries[n].position, entries[n].velocity] for n in NIDS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite hand feedback")
    age = now_wall - int(frame.header.timestamp_us) / 1e6
    if not 0 <= age <= 0.15:
        raise ValueError("Hand feedback is stale or from a future clock")
    return values[:, 0], values[:, 1], now_mono - age


class HandOwner:
    def __init__(self, sdk, side, address):
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
        except BaseException:
            self.close_readonly()
            raise

    def poll(self, now_mono, now_wall):
        # Bound draining work so a producer cannot starve stop processing.
        started = time.monotonic()
        newest_state = newest_diagnostics = None
        for _ in range(256):
            frame = self.state_sub.recv()
            if frame is None:
                break
            newest_state = frame
        for _ in range(64):
            frame = self.diagnostic_sub.recv()
            if frame is None:
                break
            by_id = {int(j.nid): j for j in frame.joints}
            if len(frame.joints) != 20 or set(by_id) != set(NIDS):
                raise ValueError("Missing or duplicate hand diagnostic joint")
            if any(int(j.error_code_current) != 0 for j in frame.joints):
                raise ValueError("Hand firmware reports a joint error")
            newest_diagnostics = (by_id, int(frame.header.timestamp_us) / 1e6)
        # A frame can arrive while draining. Compare it to the post-read clock,
        # not the earlier timestamp at function entry. Only the newest complete
        # state is decoded, while firmware errors in any drained diagnostic latch.
        elapsed = time.monotonic() - started
        now_mono, now_wall = now_mono + elapsed, now_wall + elapsed
        if newest_state is not None:
            self.latest = decode_state(newest_state, now_wall, now_mono)
            self.latest_received = now_mono
        if newest_diagnostics is not None:
            by_id, stamp = newest_diagnostics
            age = now_wall - stamp
            if not 0 <= age <= 0.15:
                raise ValueError("Stale hand diagnostics")
            self.diagnostics = (by_id, now_mono - age)
        if self.latest is None or now_mono - self.latest[2] > 0.15:
            raise ValueError("No fresh measured hand state")
        if self.diagnostics is None or now_mono - self.diagnostics[1] > 0.15:
            raise ValueError("No fresh hand diagnostics")
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
                    return
                time.sleep(0.01)
            raise RuntimeError("All 20 hand joints did not confirm enabled state")
        except BaseException:
            self.emergency_stop()
            raise

    def submit(self, positions, *, created, valid_until, now, lower, upper):
        if not self.owns_enable or self.faulted or self.last is None:
            raise RuntimeError("Hand is not owned/enabled")
        q = np.asarray(positions, dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError("Invalid hand target")
        if not created <= now < valid_until or not 0 < valid_until - created <= 0.020000001:
            raise ValueError("Hand command expired at SDK boundary")
        if np.any(q < lower) or np.any(q > upper):
            raise ValueError("Hand command exceeds position limits")
        measured, velocity, stamp = self.poll(now, time.time())
        if np.any(np.abs(velocity) > HAND_SPEED_RAD_S) or np.max(np.abs(q - measured)) > 0.05:
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
        delta = q - self.last[0]
        allowed = HAND_SPEED_RAD_S * dt
        limited = self.last[0] + np.clip(delta, -allowed, allowed)
        # Do not accidentally use this field as a velocity limiter: MIT velocity
        # is a feedforward target. The limit is enforced on q(t) above.
        if time.monotonic() >= valid_until:
            raise ValueError("Hand command expired during feedback checks")
        commands = [self.sdk.JointCommand(float(x), 0.0, 0.0) for x in limited]
        if time.monotonic() >= valid_until:
            raise ValueError("Hand command expired before publisher send")
        self.publisher.send(commands)
        # Store completion time: next frame's available interval is conservative
        # even when the publisher call itself takes time. Failed sends don't commit.
        self.last = (limited.copy(), time.monotonic())
        return {
            "positions": limited.tolist(),
            "rate_limited_indices": np.flatnonzero(np.abs(delta) > allowed).tolist(),
            "sent_at": self.last[1],
            "interval_seconds": dt,
        }

    def emergency_stop(self):
        self.faulted = True
        if self.owns_enable:
            self.hand.emergency_stop()
        # Keep the connection/feedback alive for confirmation; a returned RPC
        # is not a claim that the hand has physically stopped.

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
        if np.max(np.abs(velocity)) > 0.02:
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
