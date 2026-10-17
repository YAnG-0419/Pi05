"""Check the entire upgraded path, including real gateway math and SDK sends."""

from dataclasses import replace
from types import SimpleNamespace as S

import numpy as np
import pytest

from experiments.weight_motion_eval.oneshot import core as slow_core
from experiments.weight_motion_eval.oneshot import limits as slow_limits
from experiments.weight_motion_eval.oneshot.ros_boundary import checked_gateway_arm_speed as slow_gateway_speed
from experiments.weight_motion_eval.oneshot.test_boundaries import message
from experiments.weight_motion_eval.oneshot.test_boundaries import reference_gate

from .planner import build_plan
from .profile import planning_config
from .runtime.core import ARM
from .runtime.core import ConsumerGuard
from .runtime.core import Feedback
from .runtime.core import Frame
from .runtime.core import checked_arm_speed
from .runtime.hand import HandOwner
from .runtime.ipc import RemoteDevices
from .runtime.ipc import wire_feedback
from .runtime.qualification import SupervisedTrial
from .runtime.ros_boundary import ArmBoundary
from .runtime.ros_boundary import checked_gateway_arm_speed
from .test_medium import inputs


def feedback(now=10.0, velocity=0.0):
    dq = np.zeros(54)
    dq[ARM] = velocity
    return Feedback(np.zeros(54), dq, (now,) * 4, (now,) * 4, hand_control="slider")


@pytest.mark.parametrize("velocity", [1.68, 2.0, 2.001])
@pytest.mark.parametrize("check", ["derivative", "slew", "measured"])
def test_new_arm_ceiling_is_enforced_at_each_boundary(velocity, check):
    if check == "measured":
        if velocity <= 2:
            feedback(velocity=velocity).check(10.0, 0)
        else:
            with pytest.raises(ValueError, match="speed exceeds"):
                feedback(velocity=velocity).check(10.0, 0)
        return
    guard = ConsumerGuard(np.full(54, -2.0), np.full(54, 2.0), hand_control="slider")
    guard.arm("run", "hash", feedback(), 10.0)
    first = Frame("run", "hash", 0, 10.0, 10.02, 0.0, np.zeros(54), np.zeros(54), "playback")
    guard.validate(first, feedback(), 10.0)
    guard.commit(first)
    q, dq = np.zeros(54), np.zeros(54)
    if check == "derivative":
        dq[ARM] = velocity
    else:
        q[ARM] = velocity * 0.01
    frame = replace(first, sequence=1, created=10.01, valid_until=10.03, positions=q, velocities=dq)
    if velocity <= 2:
        guard.validate(frame, feedback(10.01), 10.01)
    else:
        with pytest.raises(ValueError, match="Command"):
            guard.validate(frame, feedback(10.01), 10.01)


@pytest.mark.parametrize("step", [0.0168, 0.02001])
def test_real_gateway_accepts_new_plan_speed_and_rejects_excess(step):
    gate, names = reference_gate()
    gate.max_joint_speed = checked_gateway_arm_speed("2.0")
    boundary = ArmBoundary(gate)
    measured = np.tile([0, 0, 0, -1, 0, 1, 0], 2).astype(float)
    torques = {"left": np.zeros(7), "right": np.zeros(7)}
    boundary.accept(message(measured, names), measured, torques, 1.0, 5_001_000_000)
    target = measured.copy()
    target[[0, 7]] += step
    if step <= 0.02:
        result = boundary.accept(message(target, names, sequence=1), measured, torques, 1.01, 5_010_000_000)
        np.testing.assert_allclose(result.positions, target)
    else:
        with pytest.raises(ValueError, match="slew protection"):
            boundary.accept(message(target, names, sequence=1), measured, torques, 1.01, 5_010_000_000)


@pytest.mark.parametrize("mode,speed", [("slider", 2.0), ("strict", np.pi / 2)])
def test_actual_sdk_send_uses_doubled_hand_limit(monkeypatch, mode, speed):
    from .runtime import hand as module

    clock = S(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    owner = HandOwner.__new__(HandOwner)
    owner.hand_control, owner.owns_enable, owner.faulted = mode, True, False
    owner.last = (np.zeros(20), 9.99)
    owner.poll = lambda *a: (owner.last[0].copy(), np.zeros(20), clock.now)
    owner.diagnostics = ({n: S(status_word=S(ext_state=2)) for n in range(20)}, 10.0)
    owner.sdk = S(JointCommand=lambda *a: a)
    sent = []
    owner.publisher = S(send=sent.append)
    for _ in range(3):
        requested = owner.last[0] + 0.04
        result = owner.submit(
            requested,
            created=clock.now,
            valid_until=clock.now + 0.02,
            now=clock.now,
            lower=np.full(20, -2.0),
            upper=np.full(20, 2.0),
        )
        clock.now += 0.01
    np.testing.assert_allclose(np.array(sent[-1])[:, 0], 3 * 0.01 * speed, atol=1e-8)
    np.testing.assert_allclose(result["positions"], 3 * 0.01 * speed, atol=1e-8)
    clock.now += 0.04
    with pytest.raises(ValueError, match="stalled"):
        owner.submit(
            owner.last[0],
            created=clock.now,
            valid_until=clock.now + 0.02,
            now=clock.now,
            lower=np.full(20, -2.0),
            upper=np.full(20, 2.0),
        )


def test_ipc_qualification_and_slow_modules_keep_distinct_limits():
    client = object.__new__(RemoteDevices)
    client.arm_speed_enforced = True
    client.arm_speed_rad_s = 2.0
    closed = []
    client.close = lambda: closed.append(True)
    assert client.decode_feedback(wire_feedback(feedback(velocity=1.68))).arm_speed_rad_s == 2.0
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        client.decode_feedback(wire_feedback(replace(feedback(), arm_speed_rad_s=1.0)))
    assert closed
    assert SupervisedTrial("slider").data["hand_velocity_rad_s"] == 2.0
    assert SupervisedTrial("strict").data["hand_velocity_rad_s"] == pytest.approx(np.pi / 2)
    assert slow_limits.SLIDER_SPEED_RAD_S == 1.0
    assert pytest.approx(np.pi / 4) == slow_limits.HAND_SPEED_RAD_S
    assert slow_core.SPEED[0] == 0.7
    with pytest.raises(ValueError):
        slow_core.checked_arm_speed(2.0)
    with pytest.raises(Exception):
        slow_gateway_speed("2.0")
    with pytest.raises(ValueError):
        checked_arm_speed(2.01)


@pytest.mark.parametrize("mode", ["slider", "strict"])
def test_both_phases_are_twice_as_fast_as_previous_medium(mode):
    # Construct the previous settings explicitly, retaining its planner's 1 s
    # approach minimum and 1 rad/s slider travel-time floor.
    config = planning_config("19999")
    config.update(hand_control=mode, continuation=True)
    old_config = dict(config, minimum_time_scale=1.25, continuation_min_approach_seconds=0.05)
    import copy

    old_config = copy.deepcopy(old_config)
    for phase in ("approach", "playback"):
        for part in ("arm", "hand"):
            for order, key in enumerate(("velocity", "acceleration", "jerk"), 1):
                old_config[phase][part][key] /= 2**order
    # Use the new planner with settings expanded back to previous time scale;
    # actual duration checks below are supplemented by historical-record checks.
    raw = np.zeros((50, 54))
    raw[:, 0] = 0.1 + 0.03 * np.sin(np.linspace(0, 6, 50))
    if mode == "strict":
        raw[:, 7] = raw[:, 0]
    limits, names = inputs()
    before = build_plan(raw, np.zeros(54), old_config, limits, names)
    after = build_plan(raw, np.zeros(54), config, limits, names)
    assert after.playback.duration == pytest.approx(before.playback.duration / 2)
    assert after.approach.duration == pytest.approx(before.approach.duration / 2)


def test_slider_initial_hand_distance_uses_two_rad_per_second():
    config = planning_config("19999")
    config.update(hand_control="slider", continuation=True)
    raw = np.zeros((50, 54))
    raw[:, 7] = 1.0
    limits, names = inputs()
    plan = build_plan(raw, np.zeros(54), config, limits, names)
    assert plan.approach.duration == pytest.approx(0.5)
