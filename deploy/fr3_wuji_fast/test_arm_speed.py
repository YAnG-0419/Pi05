"""Session speed isolation across prediction, feedback, output and gateway."""

from dataclasses import replace
import json

import numpy as np
import pytest

from deploy.fr3_wuji_fast.timeline import check_arm_speed
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import SPEED
from experiments.weight_motion_eval.oneshot.core import ConsumerGuard
from experiments.weight_motion_eval.oneshot.core import Feedback
from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.core import checked_arm_speed
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices
from experiments.weight_motion_eval.oneshot.ipc import wire_feedback
from experiments.weight_motion_eval.oneshot.ros_boundary import ArmBoundary
from experiments.weight_motion_eval.oneshot.ros_boundary import checked_gateway_arm_speed
from experiments.weight_motion_eval.oneshot.test_boundaries import message
from experiments.weight_motion_eval.oneshot.test_boundaries import reference_gate


def feedback(now=10.0, speed=1.0, velocity=0.0):
    v = np.zeros(54)
    v[ARM] = velocity
    return Feedback(np.zeros(54), v, (now,) * 4, (now,) * 4, arm_speed_rad_s=speed)


@pytest.mark.parametrize("velocity", [0.9, 1.0, 1.001])
def test_prediction_and_feedback_agree_on_one_rad_per_second(velocity):
    actions = np.zeros((50, 54))
    actions[1:, ARM] = velocity / 30
    if velocity <= 1:
        check_arm_speed(actions)
        feedback(velocity=velocity).check(10, 0)
    else:
        with pytest.raises(ValueError, match="speed exceeds"):
            check_arm_speed(actions)
        with pytest.raises(ValueError, match="speed exceeds"):
            feedback(velocity=velocity).check(10, 0)
    with pytest.raises(ValueError, match="speed exceeds"):
        replace(feedback(velocity=velocity), arm_speed_rad_s=0.7).check(10, 0)
    np.testing.assert_array_equal(SPEED[ARM], 0.7)


@pytest.mark.parametrize("invalid", [0, -1, 1.01, float("nan"), float("inf"), True])
def test_invalid_session_limits_rejected(invalid):
    with pytest.raises(ValueError, match="Arm speed"):
        checked_arm_speed(invalid)


@pytest.mark.parametrize(("speed", "velocity"), [(1.0, 0.9), (1.0, 1.0), (1.0, 1.001), (0.7, 0.9)])
@pytest.mark.parametrize("check", ["derivative", "slew"])
def test_final_consumer_enforces_its_session_limit(speed, velocity, check):
    guard = ConsumerGuard(np.full(54, -2), np.full(54, 2), arm_speed_rad_s=speed)
    guard.arm("run", "hash", feedback(speed=speed), 10)
    first = Frame("run", "hash", 0, 10, 10.02, 0, np.zeros(54), np.zeros(54), "playback")
    guard.validate(first, feedback(speed=speed), 10)
    guard.commit(first)
    q, v = np.zeros(54), np.zeros(54)
    if check == "slew":
        q[ARM] = velocity * 0.01
    else:
        v[ARM] = velocity
    frame = replace(first, sequence=1, created=10.01, valid_until=10.03, positions=q, velocities=v)
    if velocity <= speed:
        guard.validate(frame, feedback(10.01, speed), 10.01)
    else:
        with pytest.raises(ValueError, match="Command"):
            guard.validate(frame, feedback(10.01, speed), 10.01)


def test_feedback_mode_mismatch_fails_before_arming():
    guard = ConsumerGuard(np.full(54, -2), np.full(54, 2), arm_speed_rad_s=1)
    with pytest.raises(ValueError, match="mode mismatch"):
        guard.arm("run", "hash", feedback(speed=0.7), 10)
    client = object.__new__(RemoteDevices)
    client.arm_speed_rad_s = 1
    closed = []
    client.close = lambda: closed.append(True)
    assert client.decode_feedback(wire_feedback(feedback())).arm_speed_rad_s == 1
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        client.decode_feedback(wire_feedback(feedback(speed=0.7)))
    assert closed == [True]


@pytest.mark.parametrize("speed", [0.7, 1.0])
def test_actual_gateway_obeys_per_session_speed(speed):
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


def test_fast_preview_wires_speed_to_bridge_and_gateway(tmp_path, monkeypatch):
    from deploy.fr3_wuji_fast import deploy

    output = tmp_path / "preview"
    monkeypatch.setattr(deploy, "safe_output", lambda p: output)
    deploy.main(["--check"])
    assert json.loads((output / "runtime.json").read_text())["arm_speed_rad_s"] == 1
    assert json.loads((output / "fast-config.json").read_text())["device_limits_inherited"] == {
        "arm_speed_rad_s": 1,
        "gateway_arm_speed_ceiling_rad_s": 1.2,
        "arm_tracking_rad": 0.08,
        "slider_speed_rad_s": 1.0,
        "hand_current_a": 2.0,
        "kp": 8.0,
        "kd": 0.1,
        "frame_lifetime_ms": 20,
    }
    commands = json.loads((output / "commands-preview.json").read_text())
    assert commands["gateway"][-2:] == ["--arm-speed-rad-s", "1.2"]
