"""Offline tests: adversarial targets, bounds, timing, settling and isolation."""

import queue
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import HAND

from . import limits as fast_limits
from .control import run_control
from .smoothing import ArmSmoother
from .smoothing import smoothing_settings
from .test_fast import LIMITS
from .test_fast import Clock
from .test_fast import Devices
from .test_fast import chunk
from .test_fast import initial


def smoother():
    return ArmSmoother(np.zeros(54), LIMITS, 0.0, speed=1.0)


def test_reversals_and_noisy_nodes_bound_velocity_acceleration_and_jerk():
    s = smoother()
    rng = np.random.default_rng(1909)
    times = np.r_[0.0, np.cumsum(rng.uniform(0.009, 0.011, 900))]
    previous_q, previous_v, previous_a = s.position.copy(), s.velocity.copy(), s.acceleration.copy()
    previous_t = 0.0
    for n, now in enumerate(times):
        # Abrupt chunk-sized reversals plus individual noisy 30 Hz nodes.
        target = np.full(54, 0.8 if n // 100 % 2 else -0.8)
        target[ARM] += rng.uniform(-0.02, 0.02, len(ARM))
        target[HAND] = rng.uniform(-1, 1, len(HAND))
        requested_velocity = rng.uniform(-1, 1, 54)
        q, v = s.step(target, requested_velocity, now)
        dt = now - previous_t
        assert np.max(np.abs(v[ARM])) <= 1 + 1e-8
        assert np.max(np.abs(s.acceleration)) <= 3 + 1e-8
        assert np.max(np.abs(s.acceleration - previous_a)) <= 30 * dt + 1e-8
        assert np.max(np.abs(v[ARM] - previous_v)) <= 3 * dt + 1e-8
        assert np.max(np.abs(q[ARM] - previous_q)) <= dt + 1e-8
        np.testing.assert_array_equal(q[HAND], target[HAND])
        np.testing.assert_array_equal(v[HAND], requested_velocity[HAND])
        previous_q, previous_v, previous_a, previous_t = q[ARM], v[ARM], s.acceleration.copy(), now


def test_emitted_positions_have_bounded_discrete_acceleration_and_jerk():
    s = smoother()
    positions = []
    for n in range(600):
        q, _ = s.step(np.full(54, 0.4 if n < 170 else -0.4), np.zeros(54), n * 0.01)
        positions.append(q[ARM])
    positions = np.asarray(positions)
    assert np.max(np.abs(np.diff(positions, n=2, axis=0))) / 0.01**2 <= 3 + 1e-7
    assert np.max(np.abs(np.diff(positions, n=3, axis=0))) / 0.01**3 <= 30 + 1e-6
    np.testing.assert_allclose(positions[-1], -0.4, atol=1e-10)
    assert s.settled(np.full(54, -0.4))


@pytest.mark.parametrize("now", [-0.001, 0.031, float("nan"), float("inf")])
def test_bad_clock_rejected_without_changing_command(now):
    s = smoother()
    with pytest.raises(ValueError, match="clock"):
        s.step(np.zeros(54), np.zeros(54), now)
    np.testing.assert_array_equal(s.position, 0)


@pytest.mark.parametrize("value", [float("nan"), 3.0])
def test_invalid_target_cannot_enter_output(value):
    s = smoother()
    target = np.zeros(54)
    target[0] = value
    with pytest.raises(ValueError):
        s.step(target, np.zeros(54), 0.01)
    np.testing.assert_array_equal(s.position, 0)


def test_full_braking_trajectory_checked_before_next_tick():
    s = smoother()
    s.position[0] = 1.999
    s.velocity[0] = 0.5
    target = np.zeros(54)
    target[0] = 1.8
    with pytest.raises(ValueError, match="braking trajectory"):
        s.step(target, np.zeros(54), 0.001)
    assert s.position[0] == 1.999


def test_backend_failure_never_falls_back_to_raw_target():
    s = smoother()
    s.generator = SimpleNamespace(validate_input=lambda *a: True, calculate=lambda *a: s.backend.Result.Error)
    with pytest.raises(RuntimeError, match="calculation failed"):
        s.step(np.full(54, 0.2), np.zeros(54), 0.01)
    np.testing.assert_array_equal(s.position, 0)


@pytest.mark.parametrize("name", ["ARM_ACCELERATION_RAD_S2", "ARM_JERK_RAD_S3"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_limits_rejected(name, value, monkeypatch):
    monkeypatch.setattr(fast_limits, name, value)
    with pytest.raises(ValueError, match="finite and positive"):
        smoothing_settings()


def test_legacy_end_to_end_no_endpoint_snap_and_settle_waits_for_generator():
    clock = Clock()
    devices = Devices(clock, arm_motion_limits="legacy")
    raw = np.zeros((50, 54))
    raw[15:, ARM] = 0.15
    raw[35:, ARM] = -0.1
    events = []
    run_control(
        devices,
        initial(chunk(raw=raw), arm_motion_limits="legacy"),
        LIMITS,
        {"rounds": 1, "arm_motion_limits": "legacy"},
        queue.Queue(),
        events.append,
        lambda *args: None,
        lambda: False,
        clock=clock,
        sleep=clock.sleep,
    )
    q = np.array([frame.positions[ARM] for frame in devices.frames])
    assert np.max(np.abs(np.diff(q, n=2, axis=0))) / 0.01**2 <= 3 + 1e-6
    assert np.max(np.abs(np.diff(q, n=3, axis=0))) / 0.01**3 <= 30 + 1e-5
    np.testing.assert_allclose(q[-1], -0.1, atol=1e-8)
    np.testing.assert_allclose(devices.frames[-1].velocities[ARM], 0, atol=1e-6)
    assert devices.finishes == 1
    assert any(e["event"] == "arm_smoothing_sample" for e in events)


def test_default_end_to_end_replays_fast_arm_derivative_without_ruckig():
    clock = Clock()
    devices = Devices(clock)
    raw = np.zeros((50, 54))
    raw[1:, ARM] = 0.1
    events = []
    run_control(
        devices,
        initial(chunk(raw=raw)),
        LIMITS,
        {"rounds": 1},
        queue.Queue(),
        events.append,
        lambda *args: None,
        lambda: False,
        clock=clock,
        sleep=clock.sleep,
    )
    playback = [frame for frame in devices.frames if frame.phase == "playback"]
    assert max(np.max(np.abs(frame.velocities[ARM])) for frame in playback) > 1.0
    assert not any(event["event"] == "arm_smoothing_sample" for event in events)
    complete = next(event for event in events if event["event"] == "complete")
    assert complete["arm_smoothing_max_deviation_rad"] is None


def test_fast_smoothing_does_not_modify_medium_or_slow_settings(monkeypatch):
    from deploy.fr3_wuji_medium import profile as medium
    from experiments.weight_motion_eval.oneshot import limits as slow

    before = (medium.ARM_SPEED_RAD_S, slow.ARM_TRACKING_TOLERANCE_RAD)
    config = medium.planning_config()
    monkeypatch.setattr(fast_limits, "ARM_ACCELERATION_RAD_S2", 2.0)
    s = smoother()
    s.step(np.full(54, 0.2), np.zeros(54), 0.01)
    assert before == (medium.ARM_SPEED_RAD_S, slow.ARM_TRACKING_TOLERANCE_RAD)
    assert medium.planning_config() == config


def test_preview_records_disabled_motion_policy_and_keeps_existing_guidance(tmp_path, monkeypatch):
    import json

    from . import deploy

    monkeypatch.setattr(deploy, "safe_output", lambda _: tmp_path / "preview")
    deploy.main(["--check"])
    config = json.loads((tmp_path / "preview/fast-config.json").read_text())
    assert config["arm_motion_limits"] == "disabled"
    assert config["arm_smoothing"] is None
    assert config["options"]["guidance_steps"] == 3
