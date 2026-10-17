"""Fast arm motion-limit policy across prediction, feedback, output and gateway."""

from dataclasses import replace
import json

import numpy as np
import pytest

from deploy.fr3_wuji_fast.ipc import RemoteDevices
from deploy.fr3_wuji_fast.timeline import check_arm_speed
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import SPEED
from experiments.weight_motion_eval.oneshot.core import ConsumerGuard
from experiments.weight_motion_eval.oneshot.core import Feedback
from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.core import checked_arm_speed
from experiments.weight_motion_eval.oneshot.ipc import wire_feedback
from experiments.weight_motion_eval.oneshot.ros_boundary import ArmBoundary
from experiments.weight_motion_eval.oneshot.ros_boundary import checked_gateway_arm_speed
from experiments.weight_motion_eval.oneshot.test_boundaries import message
from experiments.weight_motion_eval.oneshot.test_boundaries import reference_gate


def feedback(
    now=10.0,
    *,
    speed=None,
    enforced=False,
    velocity=0.0,
    positions=None,
):
    q = np.zeros(54) if positions is None else np.asarray(positions, dtype=float)
    v = np.zeros(54)
    v[ARM] = velocity
    return Feedback(
        q,
        v,
        (now,) * 4,
        (now,) * 4,
        arm_speed_rad_s=speed,
        arm_speed_enforced=enforced,
    )


def test_disabled_policy_accepts_arm_motion_above_the_legacy_speed():
    actions = np.zeros((50, 54))
    actions[1:, ARM] = 2.5 / 30
    with pytest.raises(ValueError, match="speed exceeds"):
        check_arm_speed(actions)

    feedback(velocity=25.0).check(10, 0)
    with pytest.raises(ValueError, match="speed exceeds"):
        feedback(speed=1.0, enforced=True, velocity=1.001).check(10, 0)
    np.testing.assert_array_equal(SPEED[ARM], 0.7)


@pytest.mark.parametrize("invalid", [0, -1, 1.01, float("nan"), float("inf"), True])
def test_invalid_legacy_session_limits_rejected(invalid):
    with pytest.raises(ValueError, match="Arm speed"):
        checked_arm_speed(invalid)


@pytest.mark.parametrize(
    "build",
    [
        lambda: feedback(speed=1.0, enforced=False),
        lambda: ConsumerGuard(
            np.full(54, -2),
            np.full(54, 2),
            arm_speed_enforced=False,
            arm_speed_rad_s=1.0,
        ),
    ],
)
def test_disabled_policy_rejects_a_numeric_speed_sentinel(build):
    with pytest.raises(ValueError, match="requires arm_speed_rad_s=None"):
        build()


def test_final_consumer_disabled_policy_keeps_checks_but_not_arm_rate_checks():
    guard = ConsumerGuard(
        np.full(54, -2),
        np.full(54, 2),
        arm_speed_enforced=False,
        arm_speed_rad_s=None,
        arm_tracking_rad=0.2,
    )
    guard.arm("run", "hash", feedback(), 10)
    first = Frame("run", "hash", 0, 10, 10.02, 0, np.zeros(54), np.zeros(54), "playback")
    guard.validate(first, feedback(), 10)
    guard.commit(first)

    q, v = np.zeros(54), np.zeros(54)
    q[ARM] = 0.05  # 5 rad/s over this 10 ms frame.
    v[ARM] = 25.0
    fast = replace(first, sequence=1, created=10.01, valid_until=10.03, positions=q, velocities=v)
    guard.validate(fast, feedback(10.01, positions=q), 10.01)

    outside = replace(fast, sequence=2, created=10.02, valid_until=10.04, positions=np.full(54, 3.0))
    with pytest.raises(ValueError, match="position exceeds"):
        guard.validate(outside, feedback(10.02, positions=q), 10.02)


@pytest.mark.parametrize("check", ["derivative", "slew"])
def test_legacy_final_consumer_still_enforces_one_rad_per_second(check):
    guard = ConsumerGuard(np.full(54, -2), np.full(54, 2), arm_speed_rad_s=1.0)
    guard.arm("run", "hash", feedback(speed=1.0, enforced=True), 10)
    first = Frame("run", "hash", 0, 10, 10.02, 0, np.zeros(54), np.zeros(54), "playback")
    guard.validate(first, feedback(speed=1.0, enforced=True), 10)
    guard.commit(first)
    q, v = np.zeros(54), np.zeros(54)
    if check == "slew":
        q[ARM] = 0.02
    else:
        v[ARM] = 1.01
    frame = replace(first, sequence=1, created=10.01, valid_until=10.03, positions=q, velocities=v)
    with pytest.raises(ValueError, match="Command"):
        guard.validate(
            frame,
            feedback(10.01, speed=1.0, enforced=True, positions=q),
            10.01,
        )


def test_feedback_policy_mismatch_fails_before_arming_and_at_ipc():
    guard = ConsumerGuard(
        np.full(54, -2),
        np.full(54, 2),
        arm_speed_enforced=False,
        arm_speed_rad_s=None,
    )
    with pytest.raises(ValueError, match="mode mismatch"):
        guard.arm("run", "hash", feedback(speed=0.7, enforced=True), 10)

    client = object.__new__(RemoteDevices)
    client.arm_speed_enforced = False
    client.arm_speed_rad_s = None
    closed = []
    client.close = lambda: closed.append(True)
    assert client.decode_feedback(wire_feedback(feedback())).arm_speed_enforced is False
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        client.decode_feedback(wire_feedback(feedback(speed=0.7, enforced=True)))
    assert closed == [True]


@pytest.mark.parametrize("speed", [0.7, 1.0])
def test_legacy_gateway_obeys_per_session_speed(speed):
    gate, names = reference_gate()
    gate.max_joint_speed = speed
    boundary = ArmBoundary(gate)
    measured = np.tile([0, 0, 0, -1, 0, 1, 0], 2).astype(float)
    torques = {"left": np.zeros(7), "right": np.zeros(7)}
    boundary.accept(message(measured, names), measured, torques, 1, 5_001_000_000)
    target = measured.copy()
    target[[0, 7]] += 0.009
    if speed == 1:
        result = boundary.accept(message(target, names, sequence=1), measured, torques, 1.01, 5_010_000_000)
        np.testing.assert_allclose(result.positions, target)
    else:
        with pytest.raises(ValueError, match="slew protection"):
            boundary.accept(message(target, names, sequence=1), measured, torques, 1.01, 5_010_000_000)


def test_disabled_gateway_accepts_large_slew_but_still_rejects_contact():
    gate, names = reference_gate()
    boundary = ArmBoundary(gate, arm_speed_enforced=False)
    measured = np.tile([0, 0, 0, -1, 0, 1, 0], 2).astype(float)
    torques = {"left": np.zeros(7), "right": np.zeros(7)}
    boundary.accept(message(measured, names), measured, torques, 1, 5_001_000_000)

    target = measured.copy()
    target[[0, 7]] += 0.2
    result = boundary.accept(message(target, names, sequence=1), measured, torques, 1.01, 5_010_000_000)
    np.testing.assert_allclose(result.positions, target)

    contacted = target.copy()
    contacted[0] += 0.001
    with pytest.raises(ValueError, match="contact"):
        boundary.accept(
            message(contacted, names, sequence=2),
            target,
            {"left": np.full(7, 10.0), "right": np.zeros(7)},
            1.02,
            5_020_000_000,
        )


def test_gateway_headroom_accepts_one_rad_per_second_clock_jitter():
    gate, names = reference_gate()
    gate.max_joint_speed = checked_gateway_arm_speed("1.2")
    boundary = ArmBoundary(gate)
    measured = np.tile([0, 0, 0, -1, 0, 1, 0], 2).astype(float)
    torques = {"left": np.zeros(7), "right": np.zeros(7)}
    boundary.accept(message(measured, names), measured, torques, 1, 5_001_000_000)
    target = measured.copy()
    target[[4, 11]] += 0.010019658
    result = boundary.accept(message(target, names, sequence=1), measured, torques, 1.01, 5_010_000_000)
    np.testing.assert_allclose(result.positions, target)


@pytest.mark.parametrize("broker", ["rtg", "rtc", "serial"])
def test_fast_preview_disables_arm_dynamics_for_every_broker(tmp_path, monkeypatch, broker):
    from deploy.fr3_wuji_fast import deploy

    output = tmp_path / broker
    monkeypatch.setattr(deploy, "safe_output", lambda p: output)
    deploy.main(["--check", "--broker-mode", broker])
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["arm_speed_enforced"] is False
    assert runtime["arm_speed_rad_s"] is None
    assert runtime["arm_tracking_rad"] == 0.2

    config = json.loads((output / "fast-config.json").read_text())
    assert config["arm_motion_limits"] == "disabled"
    assert config["arm_smoothing"] is None
    assert config["device_limits_inherited"]["arm_speed_enforced"] is False
    assert config["device_limits_inherited"]["gateway_arm_speed_ceiling_rad_s"] is None
    assert config["device_limits_inherited"]["arm_tracking_rad"] == 0.2

    commands = json.loads((output / "commands-preview.json").read_text())
    assert commands["gateway"][-1] == "--disable-arm-speed-limit"
