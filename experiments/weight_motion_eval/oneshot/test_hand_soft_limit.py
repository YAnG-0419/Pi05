"""Directional caps, release hysteresis and measured settling, no hardware."""

from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as S

import numpy as np
import pytest

from experiments.weight_motion_eval.planner import build_plan
from experiments.weight_motion_eval.planner import read_config

from .core import ARM
from .core import Admission
from .core import ConsumerGuard
from .core import Feedback
from .core import Frame
from .core import OneShot
from .hand import HandOwner
from .hand_soft_limit import StallSoftLimit
from .hand_trace import HandTrace
from .ipc import wire_feedback
from .test_devices import make_session


@pytest.mark.parametrize("direction", [-1, 1])
def test_caps_both_loading_directions_and_requires_actual_retreat(direction):
    limit = StallSoftLimit()
    measured, last, target = (np.zeros(20) for _ in range(3))
    last[17], target[17] = direction * 0.1, direction * 0.5
    result, events = limit.update(target, measured, last, {17}, 10)
    assert result[17] == last[17]
    assert events[0]["measured_at_trigger"] == 0
    assert events[0]["direction"] == direction
    # Even after warning clears, pressure direction stays capped across rounds.
    result, events = limit.update(target, measured, last, set(), 11)
    assert result[17] == last[17]
    assert not events
    target[17] = -direction * 0.1
    result, _ = limit.update(target, measured, last, set(), 12)
    assert result[17] == target[17]  # Retreat is never blocked.
    assert 17 in limit.contacts  # Requested retreat alone is insufficient.
    last[17] = target[17]
    measured[17] = -direction * 0.03
    _, events = limit.update(target, measured, last, {17}, 13)
    assert not events  # Warning still active.
    _, events = limit.update(target, measured, last, set(), 14)
    assert events[0]["action"] == "released"
    assert not limit.contacts


def test_ambiguous_loading_direction_does_not_guess():
    limit = StallSoftLimit()
    with pytest.raises(ValueError, match="ambiguous"):
        limit.update(np.ones(20), np.zeros(20), np.zeros(20), {0}, 10)
    assert not limit.contacts


def test_sdk_boundary_continues_bounded_contact_and_retreat_obeys_rate_limit(monkeypatch):
    from . import hand as module
    clock = S(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    owner = HandOwner.__new__(HandOwner)
    owner.hand_control, owner.owns_enable, owner.faulted, owner.side = "slider", True, False, "right"
    q = np.zeros(20)
    q[17] = 0.1
    owner.last = (q.copy(), 9.99)
    owner.poll = lambda *args: (np.zeros(20), np.zeros(20), clock.now)
    owner.diagnostics = ({n: S(status_word=S(ext_state=2)) for n in range(20)}, 10)
    owner.sdk = S(JointCommand=lambda *args: args)
    owner.fault_status = S(stalled_nids=lambda: {22}, pending=deque(maxlen=40))
    owner.trace = HandTrace()
    sent = []
    owner.publisher = S(send=sent.append)
    def submit(target):
        return owner.submit(target, created=clock.now, valid_until=clock.now + 0.02, now=clock.now,
                            lower=np.full(20, -2), upper=np.full(20, 2))
    q[17] = 0.7
    report = submit(q)
    assert report["positions"][17] == pytest.approx(0.11)
    assert report["soft_limited_indices"] == []
    assert report["contacts"] == ((17, 1, 0.1, 10.0),)
    events = [e for e in owner.trace.before if e["event"] == "bounded_contact"]
    assert events[0]["nid"] == 22
    assert events[0]["bound"] == 0.1
    assert "NID=22" in owner.fault_status.pending[0]
    clock.now += 0.01
    q[17] = -0.5
    report = submit(q)
    assert report["positions"][17] == pytest.approx(0.10)
    assert report["rate_limited_indices"] == [17]
    assert report["soft_limited_indices"] == []


@pytest.mark.parametrize(("measured_error", "reached", "accepted"), [(0.1000503406, True, True), (0.25001, True, False), (0.05, False, False)])
def test_player_uses_clamped_target_but_requires_real_settling(measured_error, reached, accepted):
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    config["hand_control"] = "slider"
    raw = np.zeros((50, 54))
    raw[:, 7] = np.linspace(0, 1, 50)
    limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    plan = build_plan(raw, np.zeros(54), config, limits, [str(i) for i in range(54)])
    player = OneShot(plan, Admission(9.8, 9.85, 9.95, "r", "w", 0), 10)
    fb = Feedback(np.zeros(54), np.zeros(54), (10,) * 4, (10,) * 4, hand_control="slider")
    player.start(fb, 10)
    player.transition("settle_end", 10, "test")
    measured = np.zeros(54)
    measured[7] = 0.3 - measured_error
    for tick in range(61):
        now = 10 + tick * 0.01
        current = replace(fb, positions=measured, source_times=(now,) * 4, receipt_times=(now,) * 4,
                          hand_soft_limits=((7, 1, 0.3),), hand_targets_reached=reached)
        # The IPC projection must not replace the actual measured position.
        current = Feedback(**wire_feedback(current))
        assert current.positions[7] == measured[7]
        player.tick(current, now)
        if player.state == "complete":
            break
    assert (player.state == "complete") == accepted


def test_device_exports_limits_and_checks_effective_target_for_next_round_and_finish():
    session = make_session()
    session.hand_control, session.checked, session.continuous = "slider", ARM, True
    session.guard = ConsumerGuard(session.limits["lower"], session.limits["upper"], hand_control="slider")
    session.prepare("run", "hash", np.zeros(54), "disable")
    measured = np.zeros(20)
    measured[17] = 0.25
    last = measured.copy()
    last[17] = 0.3
    target = measured.copy()
    target[17] = 1
    right = session.hands["right"]
    right.soft_limit = StallSoftLimit()
    right.soft_limit.update(target, measured, last, {17}, 10)
    right.last = (last, 10)
    right.poll = lambda now, wall: (measured, np.zeros(20), now)
    q = np.zeros(54)
    q[34:54] = target
    session.last = Frame("run", "hash", 0, 10, 10.02, 0, q, np.zeros(54), "settle_end")
    fb = session.feedback()
    assert fb.hand_soft_limits == ((51, 1, 0.3),)
    assert fb.hand_targets_reached
    assert session.next_plan("run", "hash2", q)["sequence"] == 1
    assert session.finish()["state"] == "released"


@pytest.mark.parametrize("limits", [((0, 1, 0.3),), ((7.0, 1, 0.3),), ((7, 0, 0.3),), ((7, 1, float("nan")),), ((7, 1, 0.3), (7, -1, 0.2))])
def test_feedback_rejects_invalid_or_arm_soft_limits(limits):
    with pytest.raises(ValueError, match="soft limit"):
        Feedback(np.zeros(54), np.zeros(54), (10,) * 4, (10,) * 4,
                 hand_control="slider", hand_soft_limits=limits)
