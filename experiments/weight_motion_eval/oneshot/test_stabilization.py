"""Acquisition dwell accepts small transients without weakening abort guards."""

from types import SimpleNamespace

import numpy as np
import pytest

from . import hand as module
from .hand_trace import HandTrace


def make_owner(monkeypatch, sample):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "time", lambda: 1000 + clock.now)
    owner = module.HandOwner.__new__(module.HandOwner)
    owner.owns_enable, owner.faulted, owner.side = True, False, "left"
    owner.hold_start = np.zeros(20)
    owner.hold_quiet_since = owner.hold_last_stamp = owner.hold_last_sent = None
    owner.trace = HandTrace()
    owner.sdk = SimpleNamespace(JointCommand=lambda *args: args)
    sent = []
    owner.publisher = SimpleNamespace(send=sent.append)
    owner.diagnostics = ({n: SimpleNamespace(status_word=SimpleNamespace(ext_state=2)) for n in module.NIDS}, 0)
    owner.poll = lambda *args: sample(clock.now)
    return owner, clock, sent


def run(owner, clock, *, cancelled=lambda: False):
    module.stabilize_hands(
        [owner],
        clock=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.001)),
        cancelled=cancelled,
    )


def test_enable_velocity_transient_then_fresh_stable_hold(monkeypatch):
    def sample(now):
        dq = np.zeros(20)
        dq[18] = np.deg2rad(4.7337) if now < 10.15 else 0.0
        return np.zeros(20), dq, now

    owner, clock, sent = make_owner(monkeypatch, sample)
    run(owner, clock)
    assert 10.65 <= clock.now <= 10.68
    assert len(sent) >= 65
    assert all(command == [(0.0, 0.0, 0.0)] * 20 for command in sent)
    assert owner.trace.before[-1]["event"] == "stabilization_ready"


@pytest.mark.parametrize("kind", ["overspeed", "pose", "firmware", "disabled", "stale"])
def test_stabilization_aborts_without_sending_on_unsafe_feedback(monkeypatch, kind):
    def sample(now):
        if kind in {"firmware", "stale"}:
            raise ValueError(kind)
        q, dq = np.zeros(20), np.zeros(20)
        if kind == "overspeed":
            dq[0] = np.deg2rad(46)
        if kind == "pose":
            q[0] = 0.006
        return q, dq, now

    owner, clock, sent = make_owner(monkeypatch, sample)
    if kind == "disabled":
        owner.diagnostics[0][1].status_word.ext_state = 1
    with pytest.raises((ValueError, RuntimeError)):
        run(owner, clock)
    assert not sent


@pytest.mark.parametrize("kind", ["repeated_frame", "never_quiet"])
def test_stabilization_times_out_without_distinct_quiet_feedback(monkeypatch, kind):
    owner, clock, sent = make_owner(
        monkeypatch,
        lambda now: (
            np.zeros(20),
            np.full(20, 0.03 if kind == "never_quiet" else 0.0),
            10.0 if kind == "repeated_frame" else now,
        ),
    )
    with pytest.raises(RuntimeError, match="timed out"):
        run(owner, clock)
    assert sent


def test_stabilization_cancellation_and_stalled_sender(monkeypatch):
    owner, clock, sent = make_owner(monkeypatch, lambda now: (np.zeros(20), np.zeros(20), now))
    with pytest.raises(RuntimeError, match="cancelled"):
        run(owner, clock, cancelled=lambda: True)
    assert not sent
    owner.hold_initial()
    clock.now += 0.04
    with pytest.raises(RuntimeError, match="deadline"):
        owner.hold_initial()
    assert len(sent) == 1


def test_both_hands_held_even_when_first_is_not_ready():
    counts = [0, 0]
    clock = SimpleNamespace(now=0.0)

    def tick(index):
        counts[index] += 1
        return counts[index] >= 3

    module.stabilize_hands(
        [SimpleNamespace(hold_initial=lambda: tick(0)), SimpleNamespace(hold_initial=lambda: tick(1))],
        clock=lambda: clock.now,
        sleep=lambda dt: setattr(clock, "now", clock.now + dt),
    )
    assert counts == [3, 3]
