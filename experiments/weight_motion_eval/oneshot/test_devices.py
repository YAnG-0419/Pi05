"""Ownership, IPC, commissioning and acquisition-clock regression tests."""

import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.weight_motion_eval.oneshot.bridge import serve_connection
from experiments.weight_motion_eval.oneshot.core import Admission
from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.core import OneShot
from experiments.weight_motion_eval.oneshot.deploy import parse_args
from experiments.weight_motion_eval.oneshot.devices import DeviceSession
from experiments.weight_motion_eval.oneshot.ipc import MAX_PACKET
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices
from experiments.weight_motion_eval.oneshot.ipc import encode
from experiments.weight_motion_eval.oneshot.ipc import receive
from experiments.weight_motion_eval.oneshot.qualification import Qualification
from experiments.weight_motion_eval.oneshot.qualification import digest
from experiments.weight_motion_eval.oneshot.runner import run_live
from experiments.weight_motion_eval.planner import build_plan
from experiments.weight_motion_eval.planner import read_config


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


class FakeArms:
    def __init__(self, clock):
        self.clock, self.stops, self.sent = clock, 0, []

    def feedback(self, now):
        return np.zeros(14), np.zeros(14), (now, now), (now, now)

    def check_ownership(self):
        pass

    def submit(self, frame):
        self.sent.append(frame)

    def stop(self):
        self.stops += 1


class FakeHand:
    def __init__(self, clock, *, failure=False, jump=False):
        self.clock, self.failure, self.jump = clock, failure, jump
        self.owns_enable = False
        self.sent, self.stops, self.enables = [], 0, 0
        self.latest_received = clock()
        self.last = None

    def identity(self):
        return {"serial": "fake"}

    def poll(self, now, wall):
        self.latest_received = now
        q = np.full(20, 0.006 if self.jump and self.owns_enable else 0.0)
        return q, np.zeros(20), now

    def enable(self, *, qualification, cancelled=lambda: False):
        self.enables += 1
        self.owns_enable = True

    def submit(self, positions, **kwargs):
        if self.failure:
            raise RuntimeError("Injected hand send failure")
        self.sent.append(np.array(positions))

    def emergency_stop(self):
        if self.owns_enable:
            self.stops += 1

    def disable_after_stop(self, now):
        self.owns_enable = False

    def close_readonly(self):
        assert not self.owns_enable


def make_session(clock=None, *, execute=True, failure=False, jump=False):
    clock = clock or Clock()
    arms = FakeArms(clock)
    hands = {"left": FakeHand(clock), "right": FakeHand(clock, failure=failure, jump=jump)}
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0)}
    qualification = SimpleNamespace(check_hand=lambda *args: None)
    return DeviceSession(arms, hands, limits, execute=execute, qualification=qualification, clock=clock)


def frame(now=10.0, sequence=0, phase="approach"):
    return Frame("a" * 32, "hash", sequence, now, now + 0.02, 0.0, np.zeros(54), np.zeros(54), phase)


def prepare(session, policy="hold"):
    session.prepare("a" * 32, "hash", np.zeros(54), policy)


def test_readonly_cannot_enable_and_address_defaults_match_workcell():
    session = make_session(execute=False)
    with pytest.raises(RuntimeError, match="read-only"):
        prepare(session)
    assert all(hand.enables == 0 for hand in session.hands.values())
    args = parse_args([])
    assert (args.left_arm_ip, args.right_arm_ip) == ("172.16.0.2", "172.16.1.2")
    assert (args.wuji_left_address, args.wuji_right_address) == ("192.168.1.110:7447", "192.168.2.111:7447")
    assert args.execute is False


def test_enable_pose_jump_stops_both_before_first_arm_frame():
    session = make_session(jump=True)
    with pytest.raises(ValueError, match="pose"):
        prepare(session)
    assert not session.arms.sent
    assert all(hand.stops == 1 for hand in session.hands.values())


def test_cancelled_acquisition_does_not_enable_devices():
    session = make_session()
    session.cancel_requested = lambda: True
    with pytest.raises(RuntimeError, match="cancelled"):
        prepare(session)
    assert all(hand.enables == 0 for hand in session.hands.values())


def test_partial_device_send_stops_all_and_session_cannot_restart():
    session = make_session(failure=True)
    prepare(session)
    with pytest.raises(RuntimeError, match="Injected"):
        session.submit(frame())
    assert len(session.arms.sent) == 1
    assert len(session.hands["left"].sent) == 1
    assert all(hand.stops == 1 for hand in session.hands.values())
    with pytest.raises(RuntimeError, match="reused"):
        prepare(session)


def test_device_watchdog_stops_when_no_further_host_requests_arrive():
    clock = Clock()
    session = make_session(clock)
    prepare(session)
    session.submit(frame())
    clock.now = 10.021
    session.service()
    assert session.state == "stopped"
    assert "watchdog" in session.fault
    assert all(hand.stops == 1 for hand in session.hands.values())


def test_telemetry_publisher_error_cannot_be_hidden_by_next_good_sample():
    session = make_session()
    prepare(session)
    session.arms.hand_feedback_errors = {"left": "Firmware diagnostic error"}
    session.service()
    assert session.state == "stopped"
    session.arms.hand_feedback_errors.clear()
    with pytest.raises(RuntimeError, match="does not accept"):
        session.submit(frame())


@pytest.mark.parametrize("policy", ["hold", "disable"])
def test_normal_finish_policy_and_hold_monitor_timeout(policy):
    clock = Clock()
    session = make_session(clock)
    prepare(session, policy)
    session.submit(frame(phase="settle_end"))
    result = session.finish()
    assert result["state"] == ("holding" if policy == "hold" else "released")
    if policy == "hold":
        clock.now = 10.31
        session.service()
        assert session.state == "stopped"
    else:
        assert not any(hand.owns_enable for hand in session.hands.values())


def test_ipc_rejects_truncation_and_nonfinite_json():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        left.sendall(b"x" * (MAX_PACKET + 10))
        with pytest.raises(ValueError, match="Truncated"):
            receive(right)
        left.sendall(b'{"bad":NaN}')
        with pytest.raises(ValueError, match="NaN"):
            receive(right)
        with pytest.raises(ValueError, match="large"):
            encode({"data": "x" * MAX_PACKET})
    finally:
        left.close()
        right.close()


def test_real_ipc_disconnect_stops_the_device_owner():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    session = make_session(time.monotonic)
    finished = threading.Event()

    def server():
        try:
            with right:
                serve_connection(right, session, stopping=finished.is_set, pump=lambda: None)
        finally:
            finished.set()

    thread = threading.Thread(target=server)
    thread.start()
    client = RemoteDevices(None, execute=True, sock=left)
    try:
        client.call("prepare", run_id="a" * 32, plan_hash="hash", start=[0.0] * 54, finish_policy="hold")
        client.submit(frame(time.monotonic()), time.monotonic())
        client.close()
        assert finished.wait(1)
        assert session.state == "stopped"
        assert all(hand.stops == 1 for hand in session.hands.values())
    finally:
        client.close()
        finished.set()
        thread.join(1)


def test_runner_starts_clock_after_slow_device_acquisition():
    clock = Clock()
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0), "speed": np.full(54, 2.0)}
    plan = build_plan(np.full((50, 54), 0.01), np.zeros(54), config, limits, [str(i) for i in range(54)])
    player = OneShot(plan, Admission(9.8, 9.85, 9.95, "request", "checkpoint", 0), 10.0)
    session = make_session(clock)

    class Devices:
        def feedback(self, now):
            clock.now += 0.001  # Reply timestamps are newer than request time.
            return session.feedback()

        def prepare(self, player, feedback, now):
            clock.now += 2  # Enabling latency must not advance approach q(t).

        def submit(self, output, now):
            assert output.plan_elapsed == pytest.approx(0.001)
            np.testing.assert_allclose(output.positions, plan.approach.sample(output.plan_elapsed))

        def stop(self):
            pass

    count = 0

    def stopping():
        nonlocal count
        count += 1
        return count > 1

    run_live(player, Devices(), stop_requested=stopping, emit=lambda *args: None, clock=clock, sleep=lambda dt: None)
    assert player.started == pytest.approx(12.002)


def test_commissioning_requires_matching_measured_artifacts(tmp_path):
    evidence = tmp_path / "measured.json"
    evidence.write_text('{"test_fixture_only": true}')
    names = [
        "arm_tracking_and_stop",
        "left_process_loss",
        "left_network_loss",
        "left_enable_no_jump",
        "right_process_loss",
        "right_network_loss",
        "right_enable_no_jump",
    ]
    data = {
        "schema_version": 1,
        "controller_sha256": "controller",
        "hand_velocity_rad_s": np.pi / 6,
        "hand_raw_initial_delta_rad": 0.2,
        "hands": {"left": {"serial": "test"}},
        "tests": {name: {"passed": True, "evidence": "measured.json", "sha256": digest(evidence)} for name in names},
    }
    record = tmp_path / "qualification.json"
    record.write_text(json.dumps(data))
    qualification = Qualification(record, controller_sha256="controller")
    qualification.check_hand("left", {"serial": "test"})
    with pytest.raises(ValueError, match="identity"):
        qualification.check_hand("left", {"serial": "other"})
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="evidence"):
        Qualification(record, controller_sha256="controller")
    with pytest.raises(ValueError, match="requires"):
        Qualification(None, controller_sha256="controller")
