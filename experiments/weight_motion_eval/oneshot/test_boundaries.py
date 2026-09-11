"""Real boundary code against fake devices plus compiled C++ deadline checks."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.hand import NIDS
from experiments.weight_motion_eval.oneshot.hand import HandOwner
from experiments.weight_motion_eval.oneshot.hand import decode_state
from experiments.weight_motion_eval.oneshot.ros_boundary import ArmBoundary
from experiments.weight_motion_eval.oneshot.transport import ros_header
from experiments.weight_motion_eval.oneshot.transport import validate_header


def test_ros_deadline_roundtrip_preserves_origin_and_rejects_queue_delay():
    frame = Frame("a" * 32, "hash", 0, 1.0, 1.02, 0, np.zeros(54), np.zeros(54), "approach")
    stamp, tag = ros_header(frame, 1.005, 5_005_000_000)
    assert stamp == 5_000_000_000
    assert validate_header(tag, stamp, 5_010_000_000) == ("a" * 32, 0, 5_020_000_000)
    with pytest.raises(ValueError, match="Expired"):
        validate_header(tag, stamp, 5_020_000_000)
    with pytest.raises(ValueError, match="20 ms"):
        validate_header(tag, stamp - 1_000_000, 5_010_000_000)
    with pytest.raises(ValueError, match="expired"):
        ros_header(frame, 1.03, 5_030_000_000)


def test_native_final_consumer_guard_matches_transport(tmp_path):
    header = Path(__file__).with_name("deadline.hpp")
    source = tmp_path / "deadline_test.cpp"
    source.write_text("""#include <cassert>
#include "deadline.hpp"
int main() {
  pi05::Envelope e;
  assert(pi05::decode("pi05v1/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/3/5020000000", e));
  assert(e.sequence == 3 && e.expiry == 5020000000LL);
  assert(pi05::fresh(5000000000LL,e.expiry,5010000000LL));
  assert(!pi05::fresh(5000000000LL,e.expiry,5020000000LL));
  assert(!pi05::fresh(5000000000LL,e.expiry,4999999999LL));
  assert(!pi05::fresh(4900000000LL,e.expiry,5010000000LL));
  assert(!pi05::decode("pi05v1/short/3/5020000000",e));
  assert(!pi05::decode("pi05v1/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/-1/5020000000",e));
  assert(!pi05::decode("pi05v1/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/0/99999999999999999999",e));
  assert(!pi05::decode("pi05v1/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/0/5020000000/trailing",e));
}""")
    executable = tmp_path / "deadline_test"
    subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Werror", "-I", str(header.parent), str(source), "-o", str(executable)],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def message(positions, names, *, sequence=0):
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=5, nanosec=0), frame_id=f"pi05v1/{'a' * 32}/{sequence}/5020000000"
        ),
        source="openpi",
        session_id="a" * 32,
        sequence=sequence,
        active_sides=["left", "right"],
        joint_names=names,
        positions=positions,
    )


def reference_gate():
    sys.path.insert(0, "/home/user/lpy/gello-retarget/ros_ws/src/teleop_core")
    from teleop_core.contract import COMMAND_JOINT_NAMES
    from teleop_core.safety import CommandSafetyGate

    gate = CommandSafetyGate(0.7, 0.05, 0.01, [6, 6, 6, 6, 3, 3, 3])
    return gate, COMMAND_JOINT_NAMES


def test_dual_arm_gateway_never_forwards_partial_acceptance():
    gate, names = reference_gate()
    boundary = ArmBoundary(gate)
    measured = np.tile([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0], 2)
    target = measured.copy()
    target[7] = 0.1
    with pytest.raises(ValueError, match="part"):
        boundary.accept(message(target, names), measured, {"left": np.zeros(7), "right": np.zeros(7)}, 1, 5_001_000_000)
    assert not any(gate.active.values())
    with pytest.raises(ValueError, match="latched"):
        boundary.accept(
            message(measured, names), measured, {"left": np.zeros(7), "right": np.zeros(7)}, 1, 5_001_000_000
        )


def test_dual_arm_gateway_preserves_contact_protection():
    gate, names = reference_gate()
    boundary = ArmBoundary(gate)
    measured = np.tile([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0], 2)
    boundary.accept(message(measured, names), measured, {"left": np.zeros(7), "right": np.zeros(7)}, 1, 5_001_000_000)
    target = measured.copy()
    target[0] += 0.001
    with pytest.raises(ValueError, match="contact"):
        boundary.accept(
            message(target, names, sequence=1),
            measured,
            {"left": np.full(7, 10), "right": np.zeros(7)},
            1.01,
            5_010_000_000,
        )


def state_frame(nids=NIDS):
    return SimpleNamespace(
        header=SimpleNamespace(timestamp_us=10_000_000),
        num_joints=20,
        joints=[SimpleNamespace(nid=n, position=i / 100, velocity=0) for i, n in enumerate(nids)],
    )


def test_hand_mapping_and_freshness_do_not_use_array_order():
    frame = state_frame()
    frame.joints.reverse()
    q, v, stamp = decode_state(frame, 10.02, 3)
    np.testing.assert_allclose(q, np.arange(20) / 100)
    assert stamp == pytest.approx(2.98)
    frame.joints[0].nid = frame.joints[1].nid
    with pytest.raises(ValueError, match="20 firmware"):
        decode_state(frame, 10.02, 3)
    with pytest.raises(ValueError, match="stale"):
        decode_state(state_frame(), 10.16, 3)


def test_hand_frame_arriving_during_drain_uses_post_read_clock(monkeypatch):
    from experiments.weight_motion_eval.oneshot import hand as module

    frame = state_frame()
    frame.header.timestamp_us = 10_005_000
    diag = SimpleNamespace(header=frame.header, joints=[SimpleNamespace(nid=n, error_code_current=0) for n in NIDS])
    states, diagnostics = iter([frame, None]), iter([diag, None])
    owner = HandOwner.__new__(HandOwner)
    owner.latest = owner.diagnostics = None
    owner.state_sub = SimpleNamespace(recv=lambda: next(states))
    owner.diagnostic_sub = SimpleNamespace(recv=lambda: next(diagnostics))
    clock = iter([1.0, 1.01])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    _, _, stamp = owner.poll(1.0, 10.0)
    assert stamp == pytest.approx(1.005)
    assert owner.latest_received == pytest.approx(1.01)


class FakeHand:
    def __init__(self):
        self.events = []

    def handedness(self):
        return SimpleNamespace(get=lambda: "left")

    def online_joints_count(self):
        return SimpleNamespace(get=lambda: 20)

    def mit_params(self):
        return SimpleNamespace(get=lambda: [SimpleNamespace(kp=8, kd=0.1)] * 20)

    def effort_limit(self):
        return SimpleNamespace(get=lambda: [1.0] * 20)

    def joint_states(self):
        return SimpleNamespace(
            subscribe=lambda: SimpleNamespace(recv=lambda: None, close=lambda: self.events.append("close_state"))
        )

    def joint_diagnostics(self):
        return SimpleNamespace(
            subscribe=lambda: SimpleNamespace(recv=lambda: None, close=lambda: self.events.append("close_diag"))
        )

    def enable(self):
        self.events.append("enable")

    def disable(self):
        self.events.append("disable")

    def disconnect(self):
        self.events.append("disconnect")


def test_hand_readonly_lifecycle_never_writes_or_enables():
    hand = FakeHand()
    sdk = SimpleNamespace(
        SdkManager=SimpleNamespace(instance=lambda: SimpleNamespace(connect=lambda **kw: hand)),
        ConnectOptions=lambda **kw: kw,
    )
    owner = HandOwner(sdk, "left", "fake")
    with pytest.raises(RuntimeError, match="not been measured"):
        owner.enable()
    owner.close_readonly()
    assert hand.events == ["close_state", "close_diag", "disconnect"]


def test_supervised_trial_binds_live_identity_and_primes_disabled_hand():
    from experiments.weight_motion_eval.oneshot.qualification import SupervisedTrial

    identities = {
        side: {
            "serial": side,
            "info": "test device",
            "mit_gains": [{"kp": 8.0, "kd": 0.1}] * 20,
            "effort_limits": [1.0] * 20,
        }
        for side in ("left", "right")
    }
    trial = SupervisedTrial()
    with pytest.raises(ValueError, match="readback"):
        trial.check_hand("left", identities["left"])
    trial.bind_hands(identities)
    hand = FakeHand()
    sdk = SimpleNamespace(
        SdkManager=SimpleNamespace(instance=lambda: SimpleNamespace(connect=lambda **kw: hand)),
        ConnectOptions=lambda **kw: kw,
        JointCommand=lambda *args: args,
    )
    owner = HandOwner(sdk, "left", "fake")
    owner.identity = lambda: identities["left"]
    owner.poll = lambda *args: (np.zeros(20), np.zeros(20), 0.0)
    diagnostic = {n: SimpleNamespace(status_word=SimpleNamespace(ext_state=1)) for n in NIDS}
    owner.diagnostics = (diagnostic, 0.0)

    def send(commands):
        assert all(j.status_word.ext_state == 1 for j in diagnostic.values())
        assert commands == [(0.0, 0.0, 0.0)] * 20
        hand.events.append("prime")

    def enable():
        hand.events.append("enable")
        for joint in diagnostic.values():
            joint.status_word.ext_state = 2

    hand.joint_command = lambda: SimpleNamespace(publish=lambda: SimpleNamespace(send=send))
    hand.enable = enable
    owner.enable(qualification=trial)
    assert hand.events == ["prime", "enable"]
    assert owner.owns_enable is True
    identities["left"]["effort_limits"] = [2.0] * 20
    with pytest.raises(ValueError, match="changed"):
        trial.check_hand("left", identities["left"])


@pytest.mark.parametrize("fault", [None, "actual_speed", "expired", "expires_during_poll", "joint_disabled", "stalled"])
def test_hand_final_send_checks_actual_and_commanded_motion(monkeypatch, fault):
    from experiments.weight_motion_eval.oneshot import hand as module

    owner = HandOwner.__new__(HandOwner)
    owner.owns_enable, owner.faulted = True, False
    owner.last = (np.zeros(20), 10.0)
    if fault == "stalled":
        owner.last = (np.zeros(20), 9.97)
    owner.sdk = SimpleNamespace(JointCommand=lambda *args: args)
    sent = []
    owner.publisher = SimpleNamespace(send=sent.append)
    diagnostics = {n: SimpleNamespace(status_word=SimpleNamespace(ext_state=2)) for n in NIDS}
    if fault == "joint_disabled":
        diagnostics[NIDS[0]].status_word.ext_state = 1
    owner.diagnostics = (diagnostics, 10.01)
    velocity = np.full(20, np.deg2rad(31) if fault == "actual_speed" else 0.0)
    owner.poll = lambda *args: (np.zeros(20), velocity, 10.01)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.04 if fault == "expires_during_poll" else 10.01)
    target = np.full(20, 0.005)
    kwargs = {
        "created": 10.01,
        "valid_until": 10.03,
        "now": 10.04 if fault == "expired" else 10.01,
        "lower": np.full(20, -1.0),
        "upper": np.full(20, 1.0),
    }
    if fault:
        with pytest.raises(ValueError, match="Hand"):
            owner.submit(target, **kwargs)
        assert not sent
        np.testing.assert_equal(owner.last[0], np.zeros(20))
    else:
        owner.submit(target, **kwargs)
        assert len(sent) == 1
        assert len(sent[0]) == 20
        assert all(command == (0.005, 0.0, 0.0) for command in sent[0])


def test_hand_limits_50_degree_targets_to_30_and_eventually_reaches_endpoint(monkeypatch):
    from experiments.weight_motion_eval.oneshot import hand as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    owner = HandOwner.__new__(HandOwner)
    owner.owns_enable, owner.faulted = True, False
    owner.last = (np.zeros(20), now[0])
    owner.sdk = SimpleNamespace(JointCommand=lambda *args: args)
    owner.diagnostics = ({n: SimpleNamespace(status_word=SimpleNamespace(ext_state=2)) for n in NIDS}, now[0])
    owner.poll = lambda *args: (owner.last[0].copy(), np.zeros(20), now[0])
    sent = []
    owner.publisher = SimpleNamespace(send=lambda commands: sent.append((now[0], commands)))
    signs = np.tile([1.0, -1.0], 10)
    reports = []
    for tick in range(1, 26):
        # Ramp at 50 deg/s for 100 ms, then hold the requested endpoint.
        # Alternate 5/15 ms delivery intervals to exercise the SDK clock.
        now[0] += 0.005 if tick % 2 else 0.015
        target = signs * np.deg2rad(50) * min(now[0] - 10.0, 0.1)
        original = target.copy()
        reports.append(
            owner.submit(
                target,
                created=now[0],
                valid_until=now[0] + 0.02,
                now=now[0],
                lower=np.full(20, -1),
                upper=np.full(20, 1),
            )
        )
        np.testing.assert_array_equal(target, original)
    timestamps = np.array([10.0] + [t for t, commands in sent])
    positions = np.array([np.zeros(20)] + [[q for q, dq, effort in commands] for t, commands in sent])
    speed = np.abs(np.diff(positions, axis=0)) / np.diff(timestamps)[:, None]
    assert np.all(speed <= np.deg2rad(30) + 1e-10)
    assert speed.max() == pytest.approx(np.deg2rad(30))
    assert reports[0]["rate_limited_indices"] == list(range(20))
    assert reports[-1]["rate_limited_indices"] == []
    np.testing.assert_allclose(positions[-1], signs * np.deg2rad(5), atol=1e-12)
    np.testing.assert_allclose(reports[-1]["positions"], positions[-1])
    assert all(dq == effort == 0 for t, commands in sent for q, dq, effort in commands)


def test_hand_failed_clamped_send_does_not_advance_limiter(monkeypatch):
    from experiments.weight_motion_eval.oneshot import hand as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 10.01)
    owner = HandOwner.__new__(HandOwner)
    owner.owns_enable, owner.faulted = True, False
    owner.last = (np.zeros(20), 10.0)
    owner.sdk = SimpleNamespace(JointCommand=lambda *args: args)
    owner.poll = lambda *args: (np.zeros(20), np.zeros(20), 10.01)
    owner.diagnostics = ({n: SimpleNamespace(status_word=SimpleNamespace(ext_state=2)) for n in NIDS}, 10.01)

    def fail(commands):
        raise RuntimeError("publisher failure")

    owner.publisher = SimpleNamespace(send=fail)
    with pytest.raises(RuntimeError, match="publisher failure"):
        owner.submit(
            np.full(20, np.deg2rad(0.5)),
            created=10.01,
            valid_until=10.03,
            now=10.01,
            lower=np.full(20, -1),
            upper=np.full(20, 1),
        )
    assert owner.last[1] == 10.0
    np.testing.assert_array_equal(owner.last[0], np.zeros(20))
