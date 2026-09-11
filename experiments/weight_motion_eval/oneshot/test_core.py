"""Execution-boundary tests: stale frames, movement, partial submission and replay."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.weight_motion_eval.oneshot.core import Admission
from experiments.weight_motion_eval.oneshot.core import ConsumerGuard
from experiments.weight_motion_eval.oneshot.core import Feedback
from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.core import OneShot
from experiments.weight_motion_eval.oneshot.runner import simulate
from experiments.weight_motion_eval.planner import build_plan
from experiments.weight_motion_eval.planner import read_config


@pytest.fixture
def case():
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    raw = np.tile(np.linspace(0.02, 0.05, 50)[:, None], (1, 54))
    plan = build_plan(raw, np.zeros(54), config, limits, [str(i) for i in range(54)])
    return plan, limits


def fb(now=0.2, position=None, velocity=None, epoch=0):
    return Feedback(
        np.zeros(54) if position is None else position,
        np.zeros(54) if velocity is None else velocity,
        (now,) * 4,
        (now,) * 4,
        epoch,
    )


def player(case):
    return OneShot(case[0], Admission(0, 0.05, 0.15, "r", "weights", 0), 0.2)


@pytest.mark.parametrize(
    ("inject", "expected"),
    [
        (None, "complete"),
        ("pause", "stopped"),
        ("feedback_loss", "fault"),
        ("consumer_delay", "fault"),
        ("scheduler_delay", "fault"),
        ("partial_submit", "fault"),
    ],
)
def test_one_shot_complete_and_faults_stop_output(case, inject, expected):
    report, data = simulate(*case, inject=inject)
    assert report["state"] == expected
    assert report["stop_requests"] == 1
    assert report["hardware_output"] is False
    assert len(data["command"]) > 1
    if inject:
        assert data["command_time"][-1] < 0.55


@pytest.mark.parametrize("args", [(0, 0.071, 0.15), (0, 0.05, 0.201), (0, 0.15, 0.1), (0, 0.05, float("nan"))])
def test_admission_never_renews_old_observation(args):
    with pytest.raises(ValueError, match="budget|deadline|Non-finite"):
        Admission(*args, "r", "weights", 0).check()


def test_plan_changes_start_movement_and_reuse_are_rejected(case):
    p = player(case)
    with pytest.raises(ValueError, match="Start pose"):
        p.start(fb(position=np.full(54, 0.006)), 0.2)
    with pytest.raises(ValueError, match="stationary"):
        p.start(fb(velocity=np.full(54, 0.03)), 0.2)
    with pytest.raises(ValueError, match="30 seconds"):
        p.start(fb(31), 31)
    p.start(fb(), 0.2)
    p.request_stop(0.21)
    with pytest.raises(ValueError, match="restart"):
        p.start(fb(0.22), 0.22)
    modified = player(case)
    modified.plan.playback.spline.c[0, 0] += 0.001
    with pytest.raises(ValueError, match="changed"):
        modified.start(fb(), 0.2)


def test_final_consumer_rejects_expired_foreign_or_repeated_frame(case):
    p = player(case)
    guard = ConsumerGuard(case[1]["lower"], case[1]["upper"])
    guard.arm(p.run_id, p.digest, fb(), 0.2)
    p.start(fb(), 0.2)
    first = p.tick(fb(), 0.2)
    with pytest.raises(ValueError, match="Expired"):
        guard.validate(first, fb(0.221), 0.221)
    with pytest.raises(ValueError, match="Foreign"):
        guard.validate(replace(first, run_id="other"), fb(), 0.2)
    guard.validate(first, fb(), 0.2)
    guard.commit(first)
    with pytest.raises(ValueError, match="Duplicate"):
        guard.validate(first, fb(0.201), 0.201)
    guard.stop()
    with pytest.raises(ValueError, match="not armed"):
        guard.validate(first, fb(0.201), 0.201)


@pytest.mark.parametrize("joint", [0, 7, 27, 34])
def test_output_slew_limit_rejects_jumps_even_with_zero_derivative(case, joint):
    p = player(case)
    guard = ConsumerGuard(case[1]["lower"], case[1]["upper"])
    guard.arm(p.run_id, p.digest, fb(), 0.2)
    first = Frame(p.run_id, p.digest, 0, 0.2, 0.22, 0, np.zeros(54), np.zeros(54), "approach")
    guard.validate(first, fb(), 0.2)
    guard.commit(first)
    jumped = np.zeros(54)
    jumped[joint] = 0.009
    next_frame = replace(first, sequence=1, created=0.21, valid_until=0.23, positions=jumped)
    with pytest.raises(ValueError, match="slew"):
        guard.validate(next_frame, fb(0.21), 0.21)


def test_measured_hand_speed_guard_and_clock_epoch(case):
    p = player(case)
    p.start(fb(), 0.2)
    velocities = np.zeros(54)
    velocities[7] = np.deg2rad(31)
    assert p.tick(fb(0.21, velocity=velocities), 0.21) is None
    assert p.state == "fault"
    p = player(case)
    p.start(fb(), 0.2)
    assert p.tick(fb(0.21, epoch=1), 0.21) is None
    assert p.state == "fault"


def test_speed_fault_identifies_all_offending_joints_and_signed_values():
    velocities = np.zeros(54)
    velocities[0], velocities[53] = 0.71, -0.8
    with pytest.raises(ValueError, match="Measured joint speed exceeds software ceiling") as error:
        fb(velocity=velocities).check(0.2, 0)
    message = str(error.value)
    assert "left_arm[0] action_index=0 velocity=0.710000000 rad/s limit=0.700000000 rad/s" in message
    assert "right_hand[19] action_index=53 velocity=-0.800000000 rad/s limit=0.523598776 rad/s" in message


def test_stop_confirmation_requires_continuous_stationarity(case):
    p = player(case)
    p.start(fb(), 0.2)
    p.request_stop(0.21)
    assert not p.stopped(fb(0.22), 0.22)
    assert not p.stopped(fb(0.5, velocity=np.full(54, 0.03)), 0.5)
    assert not p.stopped(fb(0.73), 0.73)
    assert p.stopped(fb(1.24), 1.24)


def test_45_degree_limit_is_in_planning_not_only_output_checks(case):
    plan, limits = case
    for phase in (plan.approach, plan.playback):
        peak = np.maximum(np.abs(phase.lower[1]), np.abs(phase.upper[1])) / phase.scale
        assert np.max(peak[np.r_[7:27, 34:54]]) <= np.deg2rad(45) + 1e-9
    config = copy.deepcopy(plan.config)
    config["playback"]["hand"]["velocity"] = 2
    raw = np.zeros((50, 54))
    raw[:, 7] = np.linspace(0, 1.0, 50)
    config["minimum_time_scale"] = 1
    config["playback"]["hand"].update(acceleration=1000, jerk=10000)
    unconstrained = build_plan(raw, np.zeros(54), config, limits, [str(i) for i in range(54)])
    with pytest.raises(ValueError, match="speed ceiling"):
        OneShot(unconstrained, Admission(0, 0.05, 0.15, "r", "w", 0), 0.2)
