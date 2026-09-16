"""UI-style hand targets must not inherit strict measured-speed interlocks."""

from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.weight_motion_eval.planner import build_plan
from experiments.weight_motion_eval.planner import read_config

from .core import ARM
from .core import HAND
from .core import Admission
from .core import ConsumerGuard
from .core import Feedback
from .core import Frame
from .core import OneShot
from .hand import HandOwner
from .hand_trace import HandTrace
from .ipc import wire_feedback
from .test_devices import make_session


def feedback(now=10.0):
    dq = np.zeros(54)
    dq[HAND] = np.deg2rad(65)
    return Feedback(np.zeros(54), dq, (now,) * 4, (now,) * 4, hand_control="slider")


def test_slider_feedback_crosses_ipc_without_old_ceiling_but_arm_still_stops():
    f = Feedback(**wire_feedback(feedback()))
    f.check(10.0, 0)
    with pytest.raises(ValueError, match="Measured"):
        replace(f, hand_control="strict").check(10.0, 0)
    dq = f.velocities.copy()
    dq[0] = 0.71
    with pytest.raises(ValueError, match="left_arm"):
        replace(f, velocities=dq).check(10.0, 0)
    with pytest.raises(ValueError, match="stale"):
        f.check(10.2, 0)


def test_slider_guard_accepts_large_hand_target_but_checks_arm_and_bounds():
    guard = ConsumerGuard(np.full(54, -2), np.full(54, 2), hand_control="slider")
    guard.arm("run", "hash", feedback(), 10.0)
    q = np.zeros(54)
    q[HAND] = 0.5
    frame = Frame("run", "hash", 0, 10.0, 10.02, 0, q, np.zeros(54), "approach")
    guard.validate(frame, feedback(), 10.0)
    q[0] = 0.081
    with pytest.raises(ValueError, match="Tracking"):
        guard.validate(replace(frame, positions=q), feedback(), 10.0)
    q[0], q[7] = 0, 2.1
    with pytest.raises(ValueError, match="position"):
        guard.validate(replace(frame, positions=q), feedback(), 10.0)


def test_sdk_slider_clamps_real_sent_positions_despite_velocity_spike(monkeypatch):
    from . import hand as module

    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    owner = HandOwner.__new__(HandOwner)
    owner.hand_control, owner.owns_enable, owner.faulted = "slider", True, False
    owner.last = (np.zeros(20), 9.99)
    owner.poll = lambda *args: (np.zeros(20), np.full(20, np.deg2rad(65)), clock.now)
    owner.diagnostics = ({n: SimpleNamespace(status_word=SimpleNamespace(ext_state=2)) for n in range(20)}, 0)
    owner.sdk = SimpleNamespace(JointCommand=lambda *args: args)
    owner.trace = HandTrace()
    sent = []
    owner.publisher = SimpleNamespace(send=sent.append)
    for i in range(30):
        clock.now = 10 + i * 0.01
        result = owner.submit(
            np.full(20, 0.5),
            created=clock.now,
            valid_until=clock.now + 0.02,
            now=clock.now,
            lower=np.full(20, -2),
            upper=np.full(20, 2),
        )
    np.testing.assert_allclose(result["positions"], 0.3, atol=1e-8)
    assert all(np.max(np.abs(np.array(b)[:, 0] - np.array(a)[:, 0])) <= 0.010000001 for a, b in pairwise(sent))
    clock.now += 0.01
    owner.diagnostics[0][0].status_word.ext_state = 1
    with pytest.raises(ValueError, match="enabled"):
        owner.submit(
            np.full(20, 0.5),
            created=clock.now,
            valid_until=clock.now + 0.02,
            now=clock.now,
            lower=np.full(20, -2),
            upper=np.full(20, 2),
        )


def test_slider_player_uses_raw_targets_and_arm_only_timing():
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    config["hand_control"] = "slider"
    limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    raw = np.zeros((50, 54))
    raw[:, HAND] = 0.5
    plan = build_plan(raw, np.zeros(54), config, limits, [str(i) for i in range(54)])
    assert plan.approach.duration < 1.01
    p = OneShot(plan, Admission(9.8, 9.85, 9.95, "r", "w", 0), 10)
    p.start(feedback(), 10)
    frame = p.tick(feedback(), 10)
    np.testing.assert_array_equal(frame.positions[HAND], 0.5)
    np.testing.assert_array_equal(frame.positions[ARM], 0)
    assert p.state == "approach"


def test_slider_device_session_skips_hand_stability_and_uses_slider_feedback():
    session = make_session(jump=True)
    session.hand_control = "slider"
    session.checked = ARM
    session.guard = ConsumerGuard(session.limits["lower"], session.limits["upper"], hand_control="slider")
    for hand in session.hands.values():
        hand.hold_initial = lambda: (_ for _ in ()).throw(AssertionError("Unexpected strict stabilization"))
    session.prepare("run", "hash", np.zeros(54), "hold")
    assert session.state == "armed"
    assert session.feedback().hand_control == "slider"


def test_checked_hand_feedback_avoids_duplicate_drain_but_still_expires(monkeypatch):
    from . import hand as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)
    owner = HandOwner.__new__(HandOwner)
    owner.hand_control, owner.owns_enable, owner.faulted = "slider", True, False
    owner.last = (np.zeros(20), 9.99)
    owner.sdk = SimpleNamespace(JointCommand=lambda *args: args)
    owner.trace = HandTrace()
    owner.hand = SimpleNamespace()
    owner.diagnostics = ({i: SimpleNamespace(status_word=SimpleNamespace(ext_state=2)) for i in range(20)}, 10.0)
    owner.publisher = SimpleNamespace(send=lambda commands: None)
    owner.poll = lambda *args: pytest.fail("Already checked streams must not be drained again")
    kwargs = dict(created=9.999, valid_until=10.019, now=10.0, lower=np.full(20, -2), upper=np.full(20, 2))
    owner.submit(np.full(20, 0.01), checked_feedback=(np.zeros(20), np.zeros(20), 10.0), **kwargs)
    with pytest.raises(ValueError, match="feedback expired"):
        owner.submit(np.zeros(20), checked_feedback=(np.zeros(20), np.zeros(20), 9.8), **kwargs)


@pytest.mark.parametrize(("phase", "error", "accepted"), [
    ("settle_start", 0.09, False),
    ("settle_end", 0.09, True),
    ("settle_end", 0.1, True),
    ("settle_end", 0.10001, False),
])
def test_only_final_hand_settling_uses_one_tenth_rad(phase, error, accepted):
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    config["hand_control"] = "slider"
    limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    plan = build_plan(np.zeros((50, 54)), np.zeros(54), config, limits, [str(i) for i in range(54)])
    player = OneShot(plan, Admission(9.8, 9.85, 9.95, "r", "w", 0), 10)
    player.start(feedback(), 10)
    player.transition(phase, 10, "test")
    q = np.zeros(54)
    q[HAND] = error
    for index in range(61):
        now = 10 + index * 0.01
        player.tick(replace(feedback(now), positions=q), now)
        if player.state != phase:
            break
    assert (player.state == ("playback" if phase == "settle_start" else "complete")) == accepted
