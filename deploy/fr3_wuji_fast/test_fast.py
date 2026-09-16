"""Pure simulation plus existing final-consumer checks; no hardware connections."""

from dataclasses import replace
import queue

import numpy as np
import pytest

from deploy.fr3_wuji_fast.control import run_control
from deploy.fr3_wuji_fast.deploy import parse_args
from deploy.fr3_wuji_fast.timeline import Timeline
from deploy.fr3_wuji_fast.timeline import check_arm_speed
from deploy.fr3_wuji_fast.timeline import checked_chunk
from deploy.fr3_wuji_fast.timeline import limit_arm_speed
from deploy.fr3_wuji_fast.timeline import prepare_initial
from deploy.fr3_wuji_fast.timeline import sample_nodes
from deploy.fr3_wuji_fast.timeline import smooth_prefix
from experiments.weight_motion_eval.oneshot import deploy as old_deploy
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import Admission
from experiments.weight_motion_eval.oneshot.core import ConsumerGuard
from experiments.weight_motion_eval.oneshot.core import Feedback

CONFIG = old_deploy.ROOT / "experiments/weight_motion_eval/config.yaml"
LIMITS = {
    "lower": np.full(54, -2.0),
    "upper": np.full(54, 2.0),
    "speed": np.full(54, 2.0),
}


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, dt):
        self.now += dt


def chunk(number=1, sent=9.85, received=9.9, raw=None):
    actions = np.tile(np.linspace(0.0, 0.01, 50)[:, None], (1, 54)) if raw is None else raw
    return checked_chunk(
        actions,
        Admission(sent - 0.02, sent, received, f"request-{number}", "weights", 0),
        number,
        LIMITS,
    )


class Devices:
    """Perfect position feedback; validate with the production final consumer."""

    def __init__(self, clock):
        self.clock = clock
        self.q = np.zeros(54)
        self.guard = ConsumerGuard(LIMITS["lower"], LIMITS["upper"], hand_control="slider", arm_speed_rad_s=1.0)
        self.frames = []
        self.prepares = self.finishes = 0
        self.last_submission = {}

    def feedback(self, now):
        return Feedback(
            self.q,
            np.zeros(54),
            (now,) * 4,
            (now,) * 4,
            hand_control="slider",
            arm_speed_rad_s=1.0,
        )

    def prepare(self, player, fb, now):
        self.prepares += 1
        self.guard.arm(player.run_id, player.digest, fb, now)

    def submit(self, frame, now):
        frame.__post_init__()
        self.guard.validate(frame, self.feedback(now), now)
        self.guard.commit(frame)
        self.q = frame.positions.copy()
        self.frames.append(frame)
        return self.feedback(now)

    def finish(self):
        assert self.frames[-1].phase == "settle_end"
        self.finishes += 1


def initial(first=None, now=10):
    return prepare_initial(first or chunk(), np.zeros(54), LIMITS, CONFIG, now)


def test_defaults_match_19999_hardware_without_changing_slow_defaults():
    before = vars(old_deploy.parse_args([]))
    args = parse_args([])
    assert not args.execute
    assert not args.read_only
    assert args.broker_mode == "rtg"
    assert args.guidance_steps == 3
    assert args.trigger_fraction == 0.5
    assert args.left_arm_ip == before["left_arm_ip"]
    assert args.right_arm_ip == before["right_arm_ip"]
    assert args.wuji_left_address == before["wuji_left_address"]
    assert args.wuji_right_address == before["wuji_right_address"]
    assert old_deploy.parse_args([]).minimum_time_scale == 2.5
    assert old_deploy.parse_args([]).continuous is False


@pytest.mark.parametrize(
    "args",
    [
        ["--rounds", "0"],
        ["--rounds", "1001"],
        ["--trigger-fraction", "nan"],
        ["--guidance-steps", "1"],
        ["--execute"],
        ["--supervised-trial"],
        ["--left-arm-ip", "wrong"],
        ["--wuji-left-address", "192.168.1.2:0"],
    ],
)
def test_invalid_options_rejected_before_any_launch(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_source_nodes_are_exactly_30_hz_without_retiming():
    actions = chunk().actions
    for n in range(50):
        q, _ = sample_nodes(actions, n / 30)
        np.testing.assert_allclose(q, actions[n], atol=1e-12)
    q, dq = sample_nodes(actions, 0.01)
    np.testing.assert_allclose(q, actions[0] * 0.7 + actions[1] * 0.3)
    np.testing.assert_allclose(dq, (actions[1] - actions[0]) * 30)
    np.testing.assert_array_equal(sample_nodes(actions, 50 / 30)[0], actions[-1])


def test_only_model_hand_outliers_project_and_raw_is_retained():
    raw = np.zeros((50, 54))
    raw[8, 7] = 3
    result = chunk(raw=raw)
    assert result.projected_hand_values == 1
    assert result.actions[8, 7] == 2
    assert raw[8, 7] == result.raw[8, 7] == 3
    raw[8, 0] = 3
    with pytest.raises(ValueError, match="Arm prediction"):
        chunk(raw=raw)


@pytest.mark.parametrize("raw", [np.zeros((50, 64)), np.full((50, 54), np.nan), np.zeros((49, 54))])
def test_foreign_model_shapes_and_nonfinite_outputs_rejected(raw):
    with pytest.raises(ValueError, match="50 x 54"):
        chunk(raw=raw)


def test_original_speed_violation_runs_at_one_rad_per_second():
    raw = np.zeros((50, 54))
    raw[1:, 0] = 0.1
    prepared = initial(chunk(raw=raw))
    assert prepared.chunk.arm_rate_limited_values > 0
    assert np.max(np.abs(np.diff(prepared.chunk.actions[:, ARM], axis=0))) == pytest.approx(1 / 30)
    check_arm_speed(prepared.chunk.actions)


def test_rate_limiter_tracks_targets_at_one_rad_per_second_and_catches_up():
    raw = np.zeros((8, 54))
    raw[1:, 6] = [0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0]
    limited, changed = limit_arm_speed(raw)
    np.testing.assert_allclose(limited[:, 6], [0, 1 / 30, 2 / 30, 0.1, 2 / 30, 1 / 30, 0, 0], atol=1e-15)
    assert changed == 4
    check_arm_speed(limited)


def test_rtg_cubic_overspeed_is_limited_instead_of_rejected():
    timeline = Timeline(chunk(), rounds=2)
    timeline.start(10)
    timeline.request(10 + 25 / 30)
    new = chunk(2, 10.84, 10.98, raw=np.full((50, 54), 0.1))
    event = timeline.install(new, 11, LIMITS)
    assert event["arm_rate_limited_values"] > 0
    check_arm_speed(timeline.actions)


def test_rtg_splice_compensates_latency_and_blends_at_actual_offset():
    t = Timeline(chunk(), rounds=2)
    t.start(10.0)
    assert not t.request_due(10.8)
    assert t.request(10 + 25 / 30)["number"] == 2
    before, _ = t.sample(11.0)
    new = chunk(2, 10.84, 10.98, raw=np.full((50, 54), 0.02))
    event = t.install(new, 11.0, LIMITS)
    assert event["latency_offset_steps"] == 5
    np.testing.assert_allclose(t.sample(11.0)[0], before)
    np.testing.assert_allclose(t.actions[2:], new.actions[7:])
    assert len(t.actions) == 45
    assert not t.request_due(11.5)


@pytest.mark.parametrize("case", ["late", "foreign", "duplicate", "old_request", "bad_splice"])
def test_rejected_splice_does_not_replace_current_actions(case):
    t = Timeline(chunk(), rounds=3)
    t.start(10.0)
    t.request(10.84)
    new = chunk(2, 10.85, 10.95, raw=np.full((50, 54), 0.02))
    now = 11.0
    if case == "late":
        now = 12.0
    elif case == "foreign":
        new.admission = replace(new.admission, checkpoint="different")
    elif case == "duplicate":
        new.number = 1
    elif case == "old_request":
        new.admission = replace(new.admission, sent_time=10.8, observation_time=10.79)
    else:
        new.actions[:] = 3.0
    old = t.actions.copy()
    with pytest.raises(ValueError, match="Unexpected|stale|speed exceeds|RTG splice"):
        t.install(new, now, LIMITS)
    assert t.number == 1
    assert t.pending_since is not None
    np.testing.assert_array_equal(t.actions, old)


def test_serial_requests_after_full_chunk_and_does_not_skip_latency_steps():
    t = Timeline(chunk(), rounds=2, mode="serial")
    t.start(10.0)
    assert not t.request_due(11.6)
    t.request(10 + 50 / 30)
    new = chunk(2, 11.68, 11.78, raw=np.full((50, 54), 0.02))
    assert t.install(new, 11.8, LIMITS)["latency_offset_steps"] == 0


def test_two_chunks_run_continuously_through_existing_guard():
    clock, incoming, events = Clock(), queue.Queue(), []
    devices = Devices(clock)
    pending = None

    def notify(event):
        nonlocal pending
        events.append(event)
        if event["event"] == "infer":
            pending = (
                clock() + 0.12,
                chunk(
                    event["number"],
                    clock() + 0.01,
                    clock() + 0.11,
                    raw=np.full((50, 54), 0.02),
                ),
            )

    def sleep(dt):
        nonlocal pending
        clock.sleep(dt)
        if pending is not None and clock() >= pending[0]:
            incoming.put_nowait(pending[1])
            pending = None

    run_control(
        devices,
        initial(),
        LIMITS,
        {"rounds": 2},
        incoming,
        notify,
        lambda *args: None,
        lambda: False,
        clock=clock,
        sleep=sleep,
    )
    assert devices.prepares == devices.finishes == 1
    assert [f.sequence for f in devices.frames] == list(range(len(devices.frames)))
    assert len({f.plan_hash for f in devices.frames}) == len({f.run_id for f in devices.frames}) == 1
    starts = [e for e in events if e["event"] == "chunk_started"]
    assert [e["number"] for e in starts] == [1, 2]
    assert starts[1]["at"] - starts[0]["at"] < 1.2
    between = [f for f in devices.frames if starts[0]["at"] <= f.created < starts[1]["at"]]
    assert all(f.phase == "playback" for f in between)
    np.testing.assert_allclose(devices.frames[-1].positions, 0.02)


def test_single_chunk_duration_is_fifty_policy_ticks():
    clock, events, devices = Clock(), [], None
    devices = Devices(clock)
    run_control(
        devices,
        initial(),
        LIMITS,
        {"rounds": 1},
        queue.Queue(),
        events.append,
        lambda *args: None,
        lambda: False,
        clock=clock,
        sleep=clock.sleep,
    )
    begin = next(e["at"] for e in events if e["event"] == "chunk_started")
    end = next(f.created for f in devices.frames if f.phase == "settle_end")
    assert 50 / 30 <= end - begin <= 50 / 30 + 0.01
    assert not any(e["event"] == "infer" for e in events)


@pytest.mark.parametrize(
    "failure",
    ["inference_loss", "overrun", "stale_feedback", "operator_stop", "device_reject"],
)
def test_control_failure_terminates_without_finishing_or_restarting(failure):
    clock, devices = Clock(), None
    devices = Devices(clock)
    original = devices.submit

    def submit(frame, now):
        if failure == "device_reject":
            raise RuntimeError("Injected gateway rejection")
        fb = original(frame, now)
        if failure == "overrun":
            clock.now += 0.011
        if failure == "stale_feedback":
            return replace(fb, source_times=(now - 0.2,) * 4)
        return fb

    devices.submit = submit

    def stopping():
        return failure == "operator_stop" and bool(devices.frames)

    with pytest.raises((RuntimeError, ValueError)):
        run_control(
            devices,
            initial(),
            LIMITS,
            {"rounds": 2},
            queue.Queue(),
            lambda e: None,
            lambda *args: None,
            stopping,
            clock=clock,
            sleep=clock.sleep,
        )
    assert devices.finishes == 0
    assert devices.prepares == 1
    if failure == "inference_loss":
        assert devices.frames[-1].phase == "playback"
        assert clock() < 20


def test_arm_speed_check_includes_splice_prefix():
    actions = np.full((50, 54), 0.1)
    smoothed = smooth_prefix(actions, np.zeros(54), 3)
    with pytest.raises(ValueError, match="speed exceeds"):
        check_arm_speed(smoothed)


def test_check_mode_never_launches_processes_or_opens_sockets(tmp_path, monkeypatch):
    from deploy.fr3_wuji_fast import deploy

    monkeypatch.setattr(deploy, "safe_output", lambda p: tmp_path / "preview")

    def forbidden(*args, **kwargs):
        raise AssertionError("check mode attempted device access")

    monkeypatch.setattr(deploy.hardware, "preflight_owners", forbidden)
    monkeypatch.setattr(deploy.hardware.Children, "launch", forbidden)
    monkeypatch.setattr(deploy, "RemoteDevices", forbidden)
    deploy.main(["--check"])
    assert (tmp_path / "preview/commands-preview.json").exists()
