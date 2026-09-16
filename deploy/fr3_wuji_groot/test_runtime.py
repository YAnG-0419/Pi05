import queue

import numpy as np
import pytest

from deploy.fr3_wuji_groot.control import run_control
from deploy.fr3_wuji_groot.timeline import Timeline
from deploy.fr3_wuji_groot.timeline import checked_chunk
from deploy.fr3_wuji_groot.timeline import prepare_initial
from experiments.weight_motion_eval.oneshot.core import Admission
from experiments.weight_motion_eval.oneshot.core import ConsumerGuard
from experiments.weight_motion_eval.oneshot.core import Feedback

LIMITS = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0), "speed": np.full(54, 2.0)}
CONFIG = "experiments/weight_motion_eval/config.yaml"


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Devices:
    def __init__(self):
        self.q = np.zeros(54)
        self.guard = ConsumerGuard(LIMITS["lower"], LIMITS["upper"], hand_control="slider")
        self.frames = []
        self.prepares = 0
        self.finishes = 0
        self.last_submission = {}

    def feedback(self, now):
        return Feedback(self.q, np.zeros(54), (now,) * 4, (now,) * 4, hand_control="slider")

    def prepare(self, player, feedback, now):
        self.prepares += 1
        self.guard.arm(player.run_id, player.digest, feedback, now)

    def submit(self, frame, now):
        self.guard.validate(frame, self.feedback(now), now)
        self.guard.commit(frame)
        self.q = frame.positions.copy()
        self.frames.append(frame)
        return self.feedback(now)

    def finish(self):
        self.finishes += 1


def chunk(number=1, *, value=0.0, sent=9.90, received=9.95):
    raw = np.full((16, 54), value)
    admission = Admission(sent - 0.02, sent, received, f"request-{number}", "a" * 64, 0)
    return checked_chunk(raw, admission, number, LIMITS)


def initial():
    return prepare_initial(chunk(), np.zeros(54), LIMITS, CONFIG, 10.0)


@pytest.mark.parametrize("value", [np.zeros((15, 54)), np.zeros((16, 53)), np.full((16, 54), np.nan)])
def test_non_groot_shapes_are_rejected(value):
    admission = Admission(9.88, 9.90, 9.95, "request", "a" * 64, 0)
    with pytest.raises(ValueError, match="16 x 54"):
        checked_chunk(value, admission, 1, LIMITS)


def test_timeline_requests_only_after_full_sixteen_step_chunk():
    timeline = Timeline(chunk(), rounds=2)
    timeline.start(10.0)
    assert not timeline.request_due(10 + 15 / 30)
    assert timeline.request(10 + 16 / 30)["number"] == 2
    replacement = chunk(2, value=0.004, sent=10.55, received=10.60)
    event = timeline.install(replacement, 10.61, LIMITS)
    assert event["latency_offset_steps"] == 0
    assert timeline.number == 2
    assert timeline.actions.shape == (16, 54)


def test_one_chunk_runs_for_sixteen_policy_ticks_through_existing_guard():
    clock = Clock()
    devices = Devices()
    events = []
    run_control(
        devices,
        initial(),
        LIMITS,
        {"rounds": 1, "guidance_steps": 3},
        queue.Queue(),
        events.append,
        lambda *args: None,
        lambda: False,
        clock=clock,
        sleep=clock.sleep,
    )
    begin = next(event["at"] for event in events if event["event"] == "chunk_started")
    end = next(frame.created for frame in devices.frames if frame.phase == "settle_end")
    assert 16 / 30 <= end - begin <= 16 / 30 + 0.01
    assert devices.prepares == devices.finishes == 1


def test_two_serial_chunks_hold_and_resume_without_bypassing_guard():
    clock = Clock()
    devices = Devices()
    incoming = queue.Queue()
    events = []

    def notify(event):
        events.append(event)
        if event["event"] == "infer":
            now = clock()
            incoming.put_nowait(chunk(2, value=0.004, sent=now, received=now))

    run_control(
        devices,
        initial(),
        LIMITS,
        {"rounds": 2, "guidance_steps": 3},
        incoming,
        notify,
        lambda *args: None,
        lambda: False,
        clock=clock,
        sleep=clock.sleep,
    )
    assert [event["number"] for event in events if event["event"] == "chunk_started"] == [1, 2]
    assert devices.prepares == devices.finishes == 1
    assert len(devices.frames) > 200
    assert [frame.sequence for frame in devices.frames] == list(range(len(devices.frames)))
