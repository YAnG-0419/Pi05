"""Setup may drain queues, but must never relabel old feedback as fresh."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ros_devices as module
from .ros_devices import ArmFeedbackUnavailable, RosArms


def owner(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "time", lambda: clock.now + 100)
    monkeypatch.setattr(module.time, "sleep", lambda dt: setattr(clock, "now", clock.now + dt))
    arms = RosArms.__new__(RosArms)
    arms.offset = 100
    arms.samples = {side: (np.zeros(7), np.zeros(7), 9.99, 9.99) for side in ("left", "right")}
    return arms, clock


def test_refresh_requires_new_receipts_on_both_arms(monkeypatch):
    arms, clock = owner(monkeypatch)
    calls = []
    def spin(budget):
        if budget == 0.005:
            calls.append(1)
            clock.now += 0.001
            side = "left" if len(calls) == 1 else "right"
            arms.samples[side] = (np.zeros(7), np.zeros(7), clock.now - 0.01, clock.now)
    arms.spin = spin
    feedback = arms.refresh_feedback()
    assert len(calls) == 2
    assert all(stamp >= 10 for stamp in feedback[3])
    assert feedback[2][1] == clock.now - 0.01  # Original source timestamp.


def test_refresh_times_out_on_missing_arm_without_restamping(monkeypatch):
    arms, clock = owner(monkeypatch)
    original = arms.samples["right"][2:]
    arms.spin = lambda budget: None
    with pytest.raises(RuntimeError, match="right: source_age=.*receipt_age="):
        arms.refresh_feedback(timeout=0.01)
    assert arms.samples["right"][2:] == original


def test_runtime_feedback_expiry_remains_immediate(monkeypatch):
    arms, clock = owner(monkeypatch)
    clock.now = 11
    arms.spin = lambda budget: None
    with pytest.raises(ArmFeedbackUnavailable, match="left: source_age=1.010000s"):
        arms.feedback(clock.now)


def test_refresh_cancellation_and_clock_fault_do_not_retry(monkeypatch):
    arms, clock = owner(monkeypatch)
    arms.spin = lambda budget: None
    with pytest.raises(RuntimeError, match="cancelled"):
        arms.refresh_feedback(cancelled=lambda: True)
    arms.offset = 99
    with pytest.raises(ValueError, match="clock offset"):
        arms.refresh_feedback()


def test_preparation_refreshes_after_blocking_enable_before_arming():
    from .test_devices import make_session, prepare

    session = make_session()
    events = []
    session.arms.refresh_feedback = lambda **kwargs: events.append("refresh")
    for side, hand in session.hands.items():
        original = hand.enable
        def enable(*, side=side, original=original, **kwargs):
            original(**kwargs)
            events.append(side + " enabled")
        hand.enable = enable
    prepare(session)
    assert session.state == "armed"
    assert events == ["refresh", "refresh", "left enabled", "refresh", "right enabled", "refresh", "refresh"]


def test_refresh_failure_after_enable_stops_acquired_devices():
    from .test_devices import make_session, prepare

    session = make_session()
    def refresh(**kwargs):
        if session.hands["left"].owns_enable:
            raise RuntimeError("Arm feedback refresh timed out")
    session.arms.refresh_feedback = refresh
    with pytest.raises(RuntimeError, match="refresh timed out"):
        prepare(session)
    assert session.state == "stopped"
    assert session.hands["left"].stops == 1
    assert session.hands["right"].enables == 0
    assert not session.arms.sent
