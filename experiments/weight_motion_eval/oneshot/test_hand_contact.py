"""Bounded contact grasp regressions; virtual clocks and fake devices only."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.weight_motion_eval.planner import build_plan, read_config

from .core import ARM, Admission, ConsumerGuard, Feedback, Frame, OneShot
from .hand_contact import BoundedContactGrasp
from .ipc import wire_feedback
from .test_devices import make_session


@pytest.mark.parametrize("direction", [-1, 1])
def test_contact_continues_model_target_and_requires_actual_retreat_to_restart(direction):
    grasp = BoundedContactGrasp()
    measured, last, target = (np.zeros(20) for _ in range(3))
    last[17], target[17] = direction * 0.1, direction * 0.7
    result, events = grasp.update(target, measured, last, {17}, 10)
    np.testing.assert_array_equal(result, target)
    assert events[0]["action"] == "engaged"
    assert grasp.snapshot()[0][3] == 10
    for now, warning in ((11, set()), (12, {17}), (30, set()), (309.999, set())):
        grasp.update(target, measured, last, warning, now)
        assert grasp.snapshot()[0][3] == 10
    with pytest.raises(ValueError, match="NID=22.*300s"):
        grasp.check_time(310)
    # A separate contact is released only when warning, command and actual
    # motion all confirm retreat (inherited directional hysteresis).
    grasp = BoundedContactGrasp()
    grasp.update(target, measured, last, {17}, 10)
    target[17] = -direction * 0.1
    grasp.update(target, measured, last, set(), 11)
    assert grasp.contacts
    last[17], measured[17] = target[17], -direction * 0.03
    _, events = grasp.update(target, measured, last, set(), 12)
    assert events[0]["action"] == "released"
    assert not grasp.contacts


def feedback(now=10, *, contacts=((51, -1, -0.07984551226775644, 10.0),)):
    measured = np.zeros(54)
    # Last measured error from deployment-f5bb94d876 / Round 8.
    measured[51] = 0.13072693347930908
    return Feedback(measured, np.zeros(54), (now,) * 4, (now,) * 4,
                    hand_control="slider", hand_contacts=contacts)


def target():
    q = np.zeros(54)
    q[51] = -0.11304361742897118
    return q


def test_round8_contact_is_settled_without_rewriting_measurement_or_model_target():
    fb = Feedback(**wire_feedback(feedback()))
    q = target()
    before = q.copy()
    assert fb.hands_settled(q, 0.1)
    assert fb.contact_indices(q) == (51,)
    assert fb.positions[51] == 0.13072693347930908
    np.testing.assert_array_equal(fb.effective_target(q), before)
    assert not replace(fb, hand_contacts=()).hands_settled(q, 0.1)
    velocity = fb.velocities.copy()
    velocity[51] = 0.021
    assert not replace(fb, velocities=velocity).hands_settled(q, 0.1)
    measured = fb.positions.copy()
    measured[35] = 0.11
    assert not replace(fb, positions=measured).hands_settled(q, 0.1)
    q[51] = 0.5  # Opening away from the diagnosed loading direction.
    assert not fb.hands_settled(q, 0.1)


def player():
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    config["hand_control"] = "slider"
    raw = np.zeros((50, 54))
    raw[:, 51] = np.linspace(0, target()[51], 50)
    limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    plan = build_plan(raw, np.zeros(54), config, limits, [str(i) for i in range(54)])
    result = OneShot(plan, Admission(9.8, 9.85, 9.95, "r", "w", 0), 10)
    result.start(feedback(), 10)
    result.transition("settle_end", 10, "test")
    return result


def test_contact_completes_after_dwell_but_never_skips_target_send():
    p = player()
    for tick in range(601):
        now = 10 + tick * 0.01
        p.tick(replace(feedback(now), hand_targets_reached=tick >= 550), now)
        if p.state == "complete":
            break
    assert p.state == "complete"
    assert p.transitions[-1]["reason"] == "final_contact_settle"
    assert now >= 16  # Includes >5 s send lag then 0.5 s measured dwell.


def test_endpoint_wait_remains_20_seconds_despite_300_second_contact_budget():
    p = player()
    for tick in range(2002):
        now = 10 + tick * 0.01
        p.tick(replace(feedback(now), hand_targets_reached=False), now)
        if p.state == "fault":
            break
    assert p.state == "fault"
    assert "Measured endpoint settling timed out" in p.reason
    assert now == pytest.approx(30.01)
    assert p.deadline == 10 + p.plan.report["total_seconds"] + 40


def test_contact_evidence_and_duration_reach_device_next_plan_finish_and_idle_watchdog():
    session = make_session()
    session.hand_control, session.checked, session.continuous = "slider", ARM, True
    session.guard = ConsumerGuard(session.limits["lower"], session.limits["upper"], hand_control="slider")
    session.prepare("run", "hash", np.zeros(54), "disable")
    for side, hand in session.hands.items():
        offset = 7 if side == "left" else 34
        hand.last = (target()[offset:offset+20].copy(), 10)
    hand = session.hands["right"]
    hand.contact_grasp = BoundedContactGrasp()
    measured = feedback().positions[34:54].copy()
    last = np.zeros(20)
    last[17] = -0.07984551226775644
    hand.contact_grasp.update(target()[34:54], measured, last, {17}, 10)
    hand.poll = lambda now, wall: (measured.copy(), np.zeros(20), now)
    session.last = Frame("run", "hash", 0, 10, 10.02, 0, target(), np.zeros(54), "settle_end")
    assert session.feedback().hand_contacts == feedback().hand_contacts
    assert session.next_plan("run", "hash2", target())["sequence"] == 1
    assert hand.contact_grasp.snapshot()[0][3] == 10  # No reset at next_plan.
    session.finish_policy = "hold"
    assert session.finish()["state"] == "holding"
    # A live hold heartbeat does not suppress the device-owned 300-second timer.
    session.clock.now = 310
    session.last_activity = 310
    session.service()
    assert session.state == "stopped"
    assert "contact duration" in session.fault


@pytest.mark.parametrize("contacts", [((0, 1, 0, 10),), ((7, 0, 0, 10),),
                                      ((7, 1, float("nan"), 10),), ((7, 1, 0, float("inf")),),
                                      ((7, 1, 0, 10), (7, -1, 0, 10))])
def test_invalid_contact_metadata_is_rejected(contacts):
    with pytest.raises(ValueError, match="contact"):
        feedback(contacts=contacts)


def test_strict_and_future_contact_metadata_are_rejected():
    with pytest.raises(ValueError, match="slider"):
        replace(feedback(), hand_control="strict")
    with pytest.raises(ValueError, match="future"):
        feedback(contacts=((51, -1, 0, 11),)).check(10, 0)


def test_sdk_checks_contact_deadline_with_cached_feedback_and_can_confirm_stop(monkeypatch):
    from collections import deque
    from types import SimpleNamespace as S
    from . import hand as module
    from .hand import HandOwner
    from .hand_trace import HandTrace

    clock = S(now=10.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    owner = HandOwner.__new__(HandOwner)
    owner.hand_control, owner.owns_enable, owner.faulted, owner.side = "slider", True, False, "right"
    owner.contact_grasp = BoundedContactGrasp()
    owner.last = (np.full(20, 0.1), 9.99)
    owner.latest = (np.zeros(20), np.zeros(20), 10)
    owner.diagnostics = ({n: S(status_word=S(ext_state=2)) for n in module.NIDS}, 10)
    owner.state_sub = owner.diagnostic_sub = S(recv=lambda: None)
    owner.sdk = S(JointCommand=lambda *args: args)
    owner.fault_status = S(stalled_nids=lambda: {22}, pending=deque(maxlen=40))
    owner.trace = HandTrace()
    sent, disabled = [], []
    owner.publisher = S(send=sent.append)
    owner.hand = S(disable=lambda: disabled.append(True))
    args = dict(lower=np.full(20, -2), upper=np.full(20, 2))
    result = owner.submit(np.full(20, 0.7), created=10, valid_until=10.02, now=10,
                          checked_feedback=owner.latest, **args)
    assert result["positions"][17] == pytest.approx(0.11)
    # Twenty seconds of the same contact remains valid at the SDK boundary.
    clock.now = 30
    owner.latest = (np.zeros(20), np.zeros(20), 30)
    owner.diagnostics = (owner.diagnostics[0], 30)
    np.testing.assert_array_equal(owner.poll(30, 30)[0], np.zeros(20))
    clock.now = 310
    owner.last = (np.full(20, 0.7), 309.99)
    owner.latest = (np.zeros(20), np.zeros(20), 310)
    owner.diagnostics = (owner.diagnostics[0], 310)
    with pytest.raises(ValueError, match="contact duration"):
        owner.submit(np.full(20, 0.7), created=310, valid_until=310.02, now=310,
                     checked_feedback=owner.latest, **args)
    assert len(sent) == 1
    with pytest.raises(ValueError, match="contact duration"):
        owner.poll(310, 310)
    owner.emergency_stop()
    assert disabled == [True]
    # Timed-out contact cannot suppress readback used to confirm stop/release.
    np.testing.assert_array_equal(owner.poll(310, 310)[0], np.zeros(20))


def test_contact_feedback_survives_20_seconds_and_expires_at_300():
    feedback(30).check(30, 0)
    feedback(309.999).check(309.999, 0)
    with pytest.raises(ValueError, match="contact duration"):
        feedback(310).check(310, 0)
