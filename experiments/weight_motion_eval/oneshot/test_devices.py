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

    def hold_initial(self):
        return True

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
        "hand_velocity_rad_s": np.pi / 4,
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


@pytest.mark.parametrize(("error", "accepted"), [(0.01544, True), (0.25, True), (0.25001, False)])
def test_device_finish_matches_arm_endpoint_tolerance(error, accepted):
    from dataclasses import replace

    session = make_session()
    prepare(session, "disable")
    session.submit(frame(phase="settle_end"))
    feedback = session.feedback()
    position = feedback.positions.copy()
    position[6] = error
    session.feedback = lambda: replace(feedback, positions=position)
    if accepted:
        assert session.finish()["state"] == "released"
    else:
        with pytest.raises(ValueError, match="not settled"):
            session.finish()


def test_ipc_timeout_disconnects_before_any_stop_rpc():
    class Socket:
        def __init__(self):
            self.closed = False
            self.sends = 0
        def settimeout(self, timeout):
            pass
        def sendall(self, packet):
            self.sends += 1
        def recvmsg(self, size):
            raise TimeoutError("simulated late reply")
        def close(self):
            self.closed = True
    sock = Socket()
    client = RemoteDevices(None, execute=True, sock=sock)
    with pytest.raises(TimeoutError, match="late reply"):
        client.call("submit")
    assert sock.closed
    client.stop()
    assert sock.sends == 1


@pytest.mark.parametrize("reject", [False, True])
def test_hand_submission_precedes_arm_ack_and_ack_failure_stops_all(reject):
    session = make_session()
    prepare(session)
    order = []
    def begin(command):
        order.append("arm published")
        return command
    def confirm(command, pending):
        assert pending is command
        assert all(hand.sent for hand in session.hands.values())
        order.append("arm ack")
        if reject:
            raise RuntimeError("Injected gateway rejection")
    session.arms.begin_submit = begin
    session.arms.confirm_submit = confirm
    if reject:
        with pytest.raises(RuntimeError, match="gateway rejection.*timings_ms"):
            session.submit(frame())
        assert session.state == "stopped"
        assert all(hand.stops == 1 for hand in session.hands.values())
        assert session.guard.last is None
    else:
        result = session.submit(frame())
        assert result["sequence"] == 0
        assert "acknowledged_ms" in result["timings_ms"]
    assert order == ["arm published", "arm ack"]


def test_submit_feedback_satisfies_health_check_but_watchdog_still_stops():
    session = make_session()
    prepare(session)
    session.submit(frame())
    def unexpected_read():
        raise AssertionError("duplicate feedback read")
    session.feedback = unexpected_read
    session.clock.now += 0.002
    session.service()
    assert session.state == "armed"
    session.clock.now += 0.02
    session.service(check_feedback=False)
    assert session.state == "stopped"
    assert "watchdog" in session.fault


def test_slow_feedback_reports_component_and_never_sends_expired_frame():
    session = make_session()
    prepare(session)
    original = session.hands["left"].poll
    def delayed(now, wall):
        session.clock.now += 0.025
        return original(session.clock(), wall)
    session.hands["left"].poll = delayed
    with pytest.raises(RuntimeError, match="Expired or future command"):
        session.submit(frame())
    assert session.feedback_timings["left_hand_ms"] == pytest.approx(25)
    assert "failed_ms" in session.fault
    assert not session.arms.sent
    assert all(not hand.sent for hand in session.hands.values())
    assert session.state == "stopped"


def test_bridge_prioritizes_queued_submit_over_idle_telemetry(monkeypatch):
    from experiments.weight_motion_eval.oneshot import bridge
    session = make_session()
    prepare(session)
    from .ipc import wire_frame
    packet = {"id": 1, "operation": "submit", "frame": wire_frame(frame())}
    monkeypatch.setattr(bridge, "receive", lambda conn: packet)
    class Socket:
        sent = None
        def settimeout(self, timeout):
            pass
        def sendall(self, data):
            self.sent = json.loads(data)
    conn = Socket()
    def idle():
        raise AssertionError("telemetry delayed queued command")
    serve_connection(conn, session, stopping=lambda: conn.sent is not None, pump=idle)
    assert conn.sent["ok"]
    assert session.last.sequence == 0


def test_bridge_keeps_first_watchdog_fault(monkeypatch):
    from experiments.weight_motion_eval.oneshot import bridge
    from experiments.weight_motion_eval.oneshot.ipc import wire_frame
    session = make_session()
    prepare(session)
    session.submit(frame())
    session.clock.now += 0.04
    request = {"id": 1, "operation": "submit", "frame": wire_frame(frame(session.clock(), sequence=1))}
    monkeypatch.setattr(bridge, "receive", lambda conn: request)
    class Socket:
        sent = None
        def settimeout(self, timeout):
            pass
        def sendall(self, data):
            self.sent = json.loads(data)
    conn = Socket()
    serve_connection(conn, session, stopping=lambda: conn.sent is not None, pump=lambda: None)
    assert session.state == "stopped"
    assert "watchdog" in session.fault
    assert not conn.sent["ok"]


@pytest.mark.parametrize("joint", [7, 40])
@pytest.mark.parametrize(("error", "accepted"), [(0.1000503406, True), (0.25, True), (0.25001, False)])
def test_hand_endpoint_tolerance_consistent_for_next_plan_and_finish(joint, error, accepted):
    from dataclasses import replace
    session = make_session()
    prepare(session, "disable")
    session.submit(frame(phase="settle_end"))
    original = session.feedback()
    measured = original.positions.copy()
    measured[joint] = error
    session.feedback = lambda: replace(original, positions=measured, hand_targets_reached=True)
    if accepted:
        session.check_endpoint()
        session.finish()
    else:
        with pytest.raises(ValueError, match="Hands"):
            session.check_endpoint()
        with pytest.raises(ValueError, match="Hands"):
            session.finish()
