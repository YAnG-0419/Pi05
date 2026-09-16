"""Fault capture and a bounded, disabled-only parameter diagnostic."""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from .analyze_hand_trace import analyze
from .hand import NIDS
from .hand import HandOwner
from .hand_response import apply_parameters
from .hand_response import excursion
from .hand_response import parse_args
from .hand_trace import HandTrace


def test_trace_preserves_faulting_frame_and_postfault_errors(tmp_path):
    trace = HandTrace(capacity=3)
    trace.add("state", value=1)
    trace.add("state", value=2)
    trace.add("state", value=3)
    trace.add("state", value=4)
    trace.trigger("overspeed")
    frame = SimpleNamespace(
        header=SimpleNamespace(timestamp_us=123, seq=7),
        num_joints=1,
        joints=[
            SimpleNamespace(
                nid=NIDS[0],
                error_code_current=8468,
                status_word=SimpleNamespace(ext_state=3, velocity_limit_active=True),
            )
        ],
    )
    trace.frame(frame, diagnostic=True)
    trace.trigger("second stop")
    path = tmp_path / "trace.jsonl"
    trace.write(path, {"serial": "fake"})
    events = [json.loads(x) for x in path.read_text().splitlines()]
    assert events[0]["dropped_before"] == 2
    assert events[-2]["event"] == "fault_trigger"
    assert events[-1]["device_timestamp_us"] == 123
    assert events[-1]["joints"][0]["error_code"] == 8468
    assert events[-1]["joints"][0]["velocity_limit_active"]


def test_all_drained_states_are_saved_before_validation_raises():
    owner = HandOwner.__new__(HandOwner)
    owner.trace = HandTrace()
    owner.latest = owner.diagnostics = None

    def state(stamp, q):
        return SimpleNamespace(
            header=SimpleNamespace(timestamp_us=stamp),
            num_joints=20,
            joints=[SimpleNamespace(nid=nid, position=q, velocity=0) for nid in NIDS],
        )

    states = iter([state(100, 0), state(200, 0.01), None])
    diagnostic = SimpleNamespace(
        header=SimpleNamespace(timestamp_us=200),
        num_joints=20,
        joints=[
            SimpleNamespace(nid=nid, error_code_current=8468, status_word=SimpleNamespace(ext_state=3)) for nid in NIDS
        ],
    )
    diagnostics = iter([diagnostic, None])
    owner.state_sub = SimpleNamespace(recv=lambda: next(states))
    owner.diagnostic_sub = SimpleNamespace(recv=lambda: next(diagnostics))
    with pytest.raises(ValueError, match="firmware"):
        owner.poll(1, 1)
    assert [e["event"] for e in owner.trace.before] == ["state", "state", "diagnostics"]
    assert owner.trace.before[-1]["joints"][0]["error_code"] == 8468
    assert owner.poll_timings["state_frames"] == 2
    assert owner.poll_timings["diagnostic_frames"] == 1
    assert owner.poll_timings["diagnostic_check_ms"] >= 0


def test_analyzer_distinguishes_velocity_spike_from_position_motion():
    events = [
        {
            "event": "state",
            "device_timestamp_us": i * 1000,
            "joints": [
                {
                    "nid": NIDS[0],
                    "position": math.radians(5) * i / 1000,
                    "velocity": math.radians(52 if i == 25 else 5),
                },
                {"nid": NIDS[1], "position": math.radians(50) * i / 1000, "velocity": math.radians(50)},
            ],
        }
        for i in range(40)
    ]
    report = analyze(events)
    spike, real = report["joints"][:2]
    assert spike["sdk_peak_deg_s"] == pytest.approx(52)
    assert spike["position_difference_peak_deg_s"] == pytest.approx(5)
    assert spike["position_20ms_peak_deg_s"] == pytest.approx(5)
    assert real["position_difference_peak_deg_s"] == pytest.approx(50)
    assert real["position_20ms_peak_deg_s"] == pytest.approx(50)


def test_single_joint_curve_is_rest_to_rest_and_bounded():
    duration, delta = 1.0, math.radians(2)
    times = np.linspace(0, 3, 3001)
    q = np.array([excursion(t, delta, duration) for t in times])
    assert q[0] == 0
    assert q[-1] == 0
    assert q.min() >= -1e-12
    assert q.max() <= delta + 1e-12
    assert np.max(np.abs(np.diff(q) / 0.001)) <= math.radians(5)


@pytest.mark.parametrize(
    "argv", [["--kp", "3"], ["--execute", "--delta-deg", "5"], ["--execute", "--speed-deg-s", "60"]]
)
def test_parameter_writes_require_execute_and_small_excursions(argv):
    with pytest.raises(SystemExit):
        parse_args(["--side", "left", *argv])


def test_parameter_change_preserves_other_joints_and_checks_disabled():
    baseline = {"mit_gains": [{"kp": 8.0, "kd": 0.1} for _ in range(20)], "effort_limits": [1.0] * 20}
    actual = copy.deepcopy(baseline)
    writes = []

    def set_gains(values):
        writes.append("gains")
        actual["mit_gains"] = [{"kp": kp, "kd": kd} for kp, kd in values]

    def set_effort(value):
        writes.append("effort")
        actual["effort_limits"][0] = value

    status = SimpleNamespace(ext_state=2)
    owner = SimpleNamespace(
        trace=HandTrace(),
        diagnostics=({0: SimpleNamespace(status_word=status)}, 0),
        hand=SimpleNamespace(
            mit_params=lambda: SimpleNamespace(set=set_gains),
            joint=lambda index: SimpleNamespace(effort_limit=lambda: SimpleNamespace(set=set_effort)),
        ),
        identity=lambda: actual,
    )
    args = SimpleNamespace(kp=4.0, kd=None, effort_a=0.5, joint=0)
    with pytest.raises(RuntimeError, match="disabled"):
        apply_parameters(owner, args, baseline)
    assert not writes
    status.ext_state = 1
    apply_parameters(owner, args, baseline)
    assert actual["mit_gains"][0]["kp"] == 4
    assert actual["mit_gains"][1:] == baseline["mit_gains"][1:]
    assert actual["effort_limits"][0] == 0.5
    assert actual["effort_limits"][1:] == baseline["effort_limits"][1:]


def test_full_simulated_response_restores_parameters_after_disable(monkeypatch, tmp_path):
    import sys

    from . import hand_response as module

    clock = SimpleNamespace(now=10.0)
    baseline = {
        "serial": "fake",
        "info": "fake firmware",
        "mit_gains": [{"kp": 8.0, "kd": 0.1} for _ in range(20)],
        "effort_limits": [1.0] * 20,
    }
    actual = copy.deepcopy(baseline)
    status = SimpleNamespace(ext_state=1)
    calls = []
    owner = SimpleNamespace(
        owns_enable=False, trace=HandTrace(), diagnostics=({0: SimpleNamespace(status_word=status)}, 0), q=np.zeros(20)
    )

    def set_gains(values):
        assert not owner.owns_enable
        actual["mit_gains"] = [{"kp": kp, "kd": kd} for kp, kd in values]

    def set_effort(value):
        assert not owner.owns_enable
        actual["effort_limits"][0] = value

    def enable(**kwargs):
        kwargs["qualification"].check_hand("left", actual)
        owner.owns_enable = True
        status.ext_state = 2
        calls.append("enable")
        owner.last = (owner.q.copy(), clock.now)

    def submit(q, **kwargs):
        calls.append("submit")
        # All joints except the selected one keep their initial targets.
        assert np.all(q[1:] == 0)
        assert abs(q[0]) <= math.radians(2) + 1e-9
        owner.q = q.copy()
        owner.last = (q.copy(), clock.now)

    def disable(now):
        owner.owns_enable = False
        status.ext_state = 1
        calls.append("disable")

    owner.identity = lambda: copy.deepcopy(actual)
    owner.configure_deployment = lambda: owner.identity()
    owner.poll = lambda *args: (owner.q.copy(), np.zeros(20), clock.now)
    owner.enable = enable
    owner.hold_initial = lambda: True
    owner.submit = submit
    owner.disable_after_stop = disable
    owner.close_readonly = lambda: calls.append("close")
    owner.hand = SimpleNamespace(
        mit_params=lambda: SimpleNamespace(set=set_gains),
        joint=lambda index: SimpleNamespace(effort_limit=lambda: SimpleNamespace(set=set_effort)),
    )
    monkeypatch.setattr(module, "HandOwner", lambda *args, **kwargs: owner)
    monkeypatch.setattr(module, "stationary", lambda *args, **kwargs: owner.q.copy())

    # A context-manager permission probe without real network access.
    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module, "socket", SimpleNamespace(AF_INET=2, SOCK_DGRAM=2, socket=lambda *args: Probe()))
    from experiments.weight_motion_eval import reference

    monkeypatch.setattr(
        reference,
        "hand_position_limits",
        lambda side: {
            "lower": np.full(20, -2.0),
            "upper": np.full(20, 2.0),
            "names": [str(i) for i in range(20)],
            "sources": {},
        },
    )
    monkeypatch.setitem(sys.modules, "wuji_sdk", SimpleNamespace())
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + max(seconds, 0.001)))
    output = tmp_path / "response"
    module.main(["--side", "left", "--execute", "--kp", "4", "--output", str(output)])
    report = json.loads((output / "response-report.json").read_text())
    assert report["state"] == "completed"
    assert report["parameters_restored"]
    assert actual == baseline
    assert calls[0] == "enable"
    assert calls[-2:] == ["disable", "close"]


@pytest.mark.parametrize(("mode", "current_limit"), [("strict", 1.0), ("slider", 2.0)])
def test_deployment_profile_is_written_disabled_and_verified(mode, current_limit):
    owner = HandOwner.__new__(HandOwner)
    owner.owns_enable = owner.faulted = False
    owner.trace = HandTrace()
    status = SimpleNamespace(ext_state=1)
    owner.diagnostics = ({n: SimpleNamespace(status_word=status) for n in NIDS}, 0)
    owner.poll = lambda *args: (np.zeros(20), np.zeros(20), 0)
    actual = {"mit_gains": [{"kp": 1.5, "kd": 0.1} for _ in NIDS], "effort_limits": [1.5] * 20}
    writes = []

    def current(value):
        writes.append("current")
        actual["effort_limits"] = [value] * 20

    def gains(values):
        writes.append("gains")
        actual["mit_gains"] = [{"kp": kp, "kd": kd} for kp, kd in values]

    owner.hand = SimpleNamespace(
        effort_limit=lambda: SimpleNamespace(set=current), mit_params=lambda: SimpleNamespace(set=gains)
    )
    owner.identity = lambda: copy.deepcopy(actual)
    owner.hand_control = mode
    result = owner.configure_deployment()
    assert writes == ["current", "gains"]
    assert result["mit_gains"] == [{"kp": 8, "kd": 0.1}] * 20
    assert result["effort_limits"] == [current_limit] * 20
    status.ext_state = 2
    with pytest.raises(RuntimeError, match="Ready"):
        owner.configure_deployment()
    assert len(writes) == 2
    status.ext_state = 1
    owner.hand.mit_params = lambda: SimpleNamespace(set=lambda values: actual["mit_gains"][0].update(kp=1.5))
    with pytest.raises(RuntimeError, match="readback mismatch"):
        owner.configure_deployment()
    assert not owner.owns_enable


@pytest.mark.parametrize("lead", [0.000031, 0.00024, 0.00099, 0.001016, 0.001999])
def test_small_device_clock_lead_is_bounded_and_does_not_future_date_feedback(monkeypatch, lead):
    from . import hand as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 3.0)
    frame = SimpleNamespace(
        header=SimpleNamespace(timestamp_us=10_000_000 + int(lead * 1e6)),
        num_joints=20,
        joints=[SimpleNamespace(nid=n, position=0.0, velocity=0.0) for n in NIDS],
    )
    assert module.decode_state(frame, 10.0, 3.0)[2] == 3.0
    diagnostic = SimpleNamespace(
        header=frame.header, num_joints=20, joints=[SimpleNamespace(nid=n, error_code_current=0) for n in NIDS]
    )
    states, diagnostics = iter([frame, None]), iter([diagnostic, None])
    owner = HandOwner.__new__(HandOwner)
    owner.latest = owner.diagnostics = None
    owner.state_sub = SimpleNamespace(recv=lambda: next(states))
    owner.diagnostic_sub = SimpleNamespace(recv=lambda: next(diagnostics))
    assert owner.poll(3.0, 10.0)[2] == 3.0
    assert owner.diagnostics[1] == 3.0
    for bad_stamp in (10_003_000, 9_849_000):
        frame.header.timestamp_us = bad_stamp
        with pytest.raises(ValueError, match="stale"):
            module.decode_state(frame, 10.0, 3.0)


def test_startup_drains_old_feedback_on_both_hands_then_admits_fresh():
    from .hand import HandFeedbackUnavailable, wait_initial_feedback

    clock = SimpleNamespace(now=0.0)
    calls = []
    attempts = {"left": 0, "right": 0}
    def poll(side):
        calls.append(side)
        attempts[side] += 1
        if attempts[side] < 3:
            raise HandFeedbackUnavailable("stale: age=0.2s")
    hands = {side: SimpleNamespace(poll=lambda *args, side=side: poll(side)) for side in attempts}
    wait_initial_feedback(hands, clock=lambda: clock.now, wall_clock=lambda: 10.0,
                          sleep=lambda dt: setattr(clock, "now", clock.now + dt), report=lambda text: None)
    assert calls == ["left", "right"] * 3


@pytest.mark.parametrize("reason", ["Hand firmware reports a joint error", "future clock", "Invalid joint IDs"])
def test_startup_does_not_retry_faults(reason):
    from .hand import wait_initial_feedback

    calls = []
    def poll(*args):
        calls.append(1)
        raise ValueError(reason)
    with pytest.raises(ValueError, match=reason):
        wait_initial_feedback({"left": SimpleNamespace(poll=poll)}, report=lambda text: None)
    assert calls == [1]


def test_startup_timeout_reports_last_feedback_age():
    from .hand import HandFeedbackUnavailable, wait_initial_feedback

    clock = SimpleNamespace(now=0.0)
    def poll(*args):
        raise HandFeedbackUnavailable("hand state: age=0.234000000s")
    with pytest.raises(RuntimeError, match="left: .*age=0.234000000s"):
        wait_initial_feedback({"left": SimpleNamespace(poll=poll)}, timeout=0.01,
                              clock=lambda: clock.now, wall_clock=lambda: 10.0,
                              sleep=lambda dt: setattr(clock, "now", clock.now + dt), report=lambda text: None)


def test_future_clock_is_not_startup_retryable():
    from .hand import HandFeedbackUnavailable, check_feedback_age

    with pytest.raises(ValueError, match="age=-1.000000000s") as caught:
        check_feedback_age(-1.0, "hand state")
    assert not isinstance(caught.value, HandFeedbackUnavailable)
    with pytest.raises(HandFeedbackUnavailable, match="age=0.200000000s"):
        check_feedback_age(0.2, "hand diagnostics")
