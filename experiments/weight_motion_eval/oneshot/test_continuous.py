"""Replanning preserves physical output continuity and fails closed on stalls."""

from dataclasses import replace
from pathlib import Path
import queue

import numpy as np
import pytest

from experiments.weight_motion_eval.planner import build_plan
from experiments.weight_motion_eval.planner import read_config

from .continuous import control_loop
from .core import Admission
from .core import ConsumerGuard
from .core import Feedback
from .core import OneShot
from .test_devices import frame
from .test_devices import make_session
from .test_devices import prepare


class Clock:
    now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make_player(start, *, continuation=False, request="first"):
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    config.update(hand_control="slider", execution_steps=30, continuation=continuation)
    raw = np.tile(start, (50, 1)) + np.linspace(0.001, 0.015, 50)[:, None]
    limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    plan = build_plan(raw, start, config, limits, [str(i) for i in range(54)])
    return OneShot(plan, Admission(9.8, 9.85, 9.9, request, "weights", 0), 10.0)


class Devices:
    def __init__(self, clock):
        self.clock = clock
        self.q = np.zeros(54)
        self.guard = ConsumerGuard(np.full(54, -2), np.full(54, 2), hand_control="slider")
        self.frames = []
        self.prepares = self.finishes = 0

    def feedback(self, now):
        return Feedback(self.q.copy(), np.zeros(54), (now,) * 4, (now,) * 4, hand_control="slider")

    def prepare(self, player, fb, now):
        self.prepares += 1
        self.guard.arm(player.run_id, player.digest, fb, now)

    def submit(self, command, now):
        self.guard.validate(command, self.feedback(now), now)
        self.guard.commit(command)
        self.q = command.positions.copy()
        self.frames.append(command)
        return self.feedback(now)

    def call(self, operation, **kwargs):
        assert operation == "next_plan"
        assert self.frames[-1].phase == "settle_end"
        np.testing.assert_allclose(kwargs["start"], self.frames[-1].positions)
        assert kwargs["run_id"] == self.guard.run_id
        self.guard.digest = kwargs["plan_hash"]
        return {"sequence": self.frames[-1].sequence + 1}

    def finish(self):
        self.finishes += 1


def test_two_chunks_hold_replan_and_keep_one_wire_session():
    clock, inbox, events = Clock(), queue.Queue(), []
    devices = Devices(clock)
    first = make_player(np.zeros(54))
    second = make_player(first.plan.raw[-1], continuation=True, request="second")
    waiting_at = None

    def notify(event):
        nonlocal waiting_at
        events.append(event)
        if event["event"] == "round_complete" and event["round"] == 1:
            waiting_at = clock()
        if event["event"] == "between" and clock() - waiting_at > 0.4 and inbox.empty():
            inbox.put(second)

    control_loop(devices, first, inbox, notify, lambda *args: None, lambda: False, 2, clock=clock, sleep=clock.sleep)
    assert devices.prepares == devices.finishes == 1
    assert len({f.run_id for f in devices.frames}) == 1
    assert [f.sequence for f in devices.frames] == list(range(len(devices.frames)))
    assert first.state == second.state == "complete"
    assert len(first.plan.raw) == len(second.plan.raw) == 30
    assert not any(t["to"] == "settle_start" for t in second.transitions)
    assert [e["round"] for e in events if e["event"] == "round_complete"] == [1, 2]
    np.testing.assert_allclose(devices.frames[-1].positions, second.plan.raw[-1])


def test_no_next_prediction_has_bounded_hold_and_no_automatic_restart():
    clock = Clock()
    devices = Devices(clock)
    with pytest.raises(RuntimeError, match="Next fresh plan"):
        control_loop(
            devices,
            make_player(np.zeros(54)),
            queue.Queue(),
            lambda event: None,
            lambda *args: None,
            lambda: False,
            2,
            clock=clock,
            sleep=clock.sleep,
            idle_timeout=0.1,
        )
    assert devices.finishes == 0
    assert devices.frames[-1].phase == "settle_end"


def test_submission_overrun_still_stops_without_burst():
    clock = Clock()
    devices = Devices(clock)
    original = devices.submit

    def slow(frame, now):
        result = original(frame, now)
        clock.now += 0.011
        return result

    devices.submit = slow
    with pytest.raises(RuntimeError, match="exceeded 100 Hz"):
        control_loop(
            devices,
            make_player(np.zeros(54)),
            queue.Queue(),
            lambda event: None,
            lambda *args: None,
            lambda: False,
            1,
            clock=clock,
            sleep=clock.sleep,
        )
    assert len(devices.frames) == 1


def test_device_rollover_rejects_stopped_sessions_and_superseded_frames():
    session = make_session()
    prepare(session)
    session.submit(frame(phase="settle_end"))
    with pytest.raises(RuntimeError, match="healthy"):
        session.next_plan("a" * 32, "next", np.zeros(54))
    session.continuous = True
    current = session.feedback()
    session.feedback = lambda: replace(current, hand_targets_reached=True)
    assert session.next_plan("a" * 32, "next", np.zeros(54))["sequence"] == 1
    with pytest.raises(ValueError, match="superseded"):
        session.guard.validate(frame(sequence=1), current, 10.0)
    session.stop()
    with pytest.raises(RuntimeError, match="healthy"):
        session.next_plan("a" * 32, "third", np.zeros(54))


def fake_bridge(sock, stop):
    """Only an in-memory robot; exercise real serialization and socket ownership."""
    import time

    from .ipc import encode
    from .ipc import receive
    from .ipc import unpack_frame
    from .ipc import wire_feedback

    device = Devices(time.monotonic)
    try:
        while not stop.is_set():
            request = receive(sock)
            op = request["operation"]
            now = time.monotonic()
            if op == "feedback":
                result = wire_feedback(device.feedback(now))
            elif op == "prepare":
                device.guard.arm(request["run_id"], request["plan_hash"], device.feedback(now), now)
                result = {"state": "armed"}
            elif op == "submit":
                frame_value = unpack_frame(request["frame"])
                fb = device.submit(frame_value, now)
                result = {"sequence": frame_value.sequence, "hands": {}, "feedback": wire_feedback(fb)}
            elif op == "next_plan":
                result = device.call(op, **{k: request[k] for k in ["run_id", "plan_hash", "start"]})
            elif op == "finish":
                result = {"state": "released"}
            elif op == "stop":
                result = {"state": "stopped", "physical_stop_confirmed": True}
            else:
                raise AssertionError(op)
            sock.sendall(encode({"id": request["id"], "ok": True, "result": result}))
    except (EOFError, OSError):
        pass
    finally:
        sock.close()


def test_spawned_controller_socket_transfer_and_two_rounds(tmp_path):
    import multiprocessing as mp
    import socket
    import threading
    import time

    from .continuous import StopFlag
    from .continuous import _worker

    clock = time.monotonic()
    first = make_player(np.zeros(54))
    second = make_player(first.plan.raw[-1], continuation=True, request="second")
    for player in (first, second):
        player.admission = Admission(clock - 0.1, clock - 0.05, clock, player.run_id, "weights", 0)
        player.prepared_at = clock
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    stop_server = threading.Event()
    server = threading.Thread(target=fake_bridge, args=(right, stop_server), daemon=True)
    server.start()
    context = mp.get_context("spawn")
    plans, status, stop = context.Queue(1), context.Queue(256), StopFlag(context)
    process = context.Process(target=_worker, args=(left, 0, "disable", first, plans, status, stop, str(tmp_path), 2))
    process.start()
    left.close()
    events = []
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            event = status.get(timeout=10)
            events.append(event)
            assert event["event"] != "failed", event
            if event["event"] == "round_complete" and event["round"] == 1:
                plans.put(second)
            if event["event"] == "complete":
                break
        assert events[-1]["event"] == "complete"
        process.join(5)
        assert process.exitcode == 0
    finally:
        stop.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(2)
        stop_server.set()
        server.join(1)


@pytest.mark.parametrize("hand_over_limit", [False, True])
def test_consumer_requests_new_observation_after_default_twenty_steps(tmp_path, hand_over_limit):
    import time
    from types import SimpleNamespace

    from openpi_client import msgpack_numpy

    from .continuous import ContinuousConsumer

    requests = []
    now = time.monotonic()
    devices = Devices(time.monotonic)
    devices.sock = SimpleNamespace(close=lambda: None)
    devices.counter, devices.finish_policy = 0, "disable"
    def prediction(**kwargs):
        actions = np.full((50, 54), 0.01 * len(requests))
        if hand_over_limit:
            actions[19, 49] = 2.16
        return msgpack_numpy.packb({"actions": actions})

    ws = SimpleNamespace(
        send=lambda packet: requests.append(msgpack_numpy.unpackb(packet)),
        recv=prediction,
    )
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    config["hand_control"] = "slider"
    consumer = ContinuousConsumer(
        devices,
        ws,
        {"checkpoint_manifest_sha256": "weights"},
        config,
        {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)},
        [str(i) for i in range(54)],
        tmp_path,
        SimpleNamespace(event=lambda event: None),
        rounds=2,
    )
    consumer.status, consumer.incoming = queue.Queue(), queue.Queue()
    process = SimpleNamespace(start=lambda: None, is_alive=lambda: True)
    consumer.context = SimpleNamespace(Process=lambda **kwargs: process)
    obs = {"observation/state": np.zeros(54)}

    def metadata():
        return {"samples": {"cam0": {"stamp": time.time() - 0.005}}}

    consumer(None, obs, metadata(), tmp_path)
    consumer(None, obs, metadata(), tmp_path)
    assert len(requests) == 1  # Playback must not trigger extra inference.
    consumer.status.put({"event": "round_complete", "round": 1})
    consumer.status.put({"event": "between", "round": 1, "target": [0.01] * 54})
    obs = {"observation/state": np.full(54, 0.011)}
    consumer(None, obs, metadata(), tmp_path)
    assert len(requests) == 2
    np.testing.assert_allclose(requests[1]["observation/state"], 0.011)
    next_player = consumer.incoming.get_nowait()
    assert next_player.plan.raw.shape == (20, 54)
    np.testing.assert_allclose(next_player.plan.start, 0.01)
    assert next_player.plan.config["continuation"]
    if hand_over_limit:
        assert next_player.plan.raw[19, 49] == 2.0
        assert next_player.plan.report["hand_prediction_projection"]["adjustment_count"] == 1
        with np.load(tmp_path / "round-0002" / "inference.npz") as saved:
            assert saved["actions"][19, 49] == 2.16
    assert next_player.admission.sent_time >= now
    consumer.status.put({"event": "round_complete", "round": 2})
    consumer.status.put({"event": "complete", "rounds": 2})
    consumer.poll()
    assert consumer.completed == 1
    assert consumer.rounds_completed == 2


def test_planning_failure_preserves_prediction_start_and_joint_error(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace as S
    from . import continuous as module
    from .continuous import ContinuousConsumer

    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(module.time, "time", lambda: clock.now)
    consumer = ContinuousConsumer.__new__(ContinuousConsumer)
    consumer.poll = lambda: None
    consumer.completed = consumer.pending = False
    consumer.process = None
    consumer.rounds_completed, consumer.rounds = 0, 10
    consumer.output = tmp_path
    consumer.model = {"checkpoint_manifest_sha256": "weights"}
    consumer.config = read_config(Path(__file__).parents[1] / "config.yaml")
    consumer.config.update(execution_steps=30, hand_control="slider")
    consumer.limits = {"lower": np.full(54, -2), "upper": np.full(54, 2), "speed": np.full(54, 2)}
    consumer.names = [f"joint{i}" for i in range(54)]
    actions = np.zeros((50, 54))
    actions[29, 51] = 2.1
    consumer.codec = S(packb=lambda obs: b"request", unpackb=lambda result: {"actions": actions})
    def receive(**kwargs):
        clock.now += 0.01
        return b"response"
    consumer.ws = S(send=lambda packet: None, recv=receive)
    consumer.devices = S(feedback=lambda now: S(positions=np.zeros(54)))
    obs = {"observation/state": np.zeros(54)}
    with pytest.raises(ValueError, match="reference joint limits"):
        consumer.consume(None, obs, {"samples": {"state": {"stamp": 10}}}, tmp_path)
    directory = tmp_path / "round-0001"
    with np.load(directory / "inference.npz") as saved:
        np.testing.assert_array_equal(saved["actions"], actions)
        np.testing.assert_array_equal(saved["recorded_start"], np.zeros(54))
    assert (directory / "admission.json").exists()
    error = json.loads((directory / "planning-error.json").read_text())
    assert error["execution_steps"] == 30
    assert "joint51" in error["error"]
    assert not (directory / "plan").exists()
