"""Real local sockets/processes with fake hardware behind the original bridge."""

import multiprocessing as mp
import queue
import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from deploy.fr3_wuji_fast.control import worker
from deploy.fr3_wuji_fast.timeline import checked_chunk
from deploy.fr3_wuji_fast.timeline import prepare_initial
from experiments.weight_motion_eval.oneshot.bridge import serve_connection
from experiments.weight_motion_eval.oneshot.continuous import StopFlag
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import Admission
from experiments.weight_motion_eval.oneshot.deploy import ROOT
from experiments.weight_motion_eval.oneshot.devices import DeviceSession
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices


class Arms:
    def __init__(self):
        self.q = np.zeros(14)
        self.stopped = False
        self.frames = []

    def feedback(self, now):
        return self.q, np.zeros(14), (now, now), (now, now)

    def check_ownership(self):
        pass

    def submit(self, frame):
        self.frames.append(frame)
        self.q = frame.positions[ARM].copy()

    def stop(self):
        self.stopped = True


class Hand:
    def __init__(self):
        self.q = np.zeros(20)
        self.last = None
        self.owns_enable = False
        self.latest_received = time.monotonic()

    def identity(self):
        return {"serial": "simulation-only"}

    def poll(self, now, wall):
        self.latest_received = now
        return self.q, np.zeros(20), now

    def enable(self, **kwargs):
        self.owns_enable = True

    def submit(self, q, **kwargs):
        self.q = np.array(q)
        self.last = (self.q.copy(), time.monotonic())
        return {"simulated": True, "position": self.q.tolist()}

    def disable_after_stop(self, now):
        self.owns_enable = False

    def emergency_stop(self):
        self.owns_enable = False


def test_spawned_rtg_worker_uses_original_device_session_and_releases(tmp_path):
    context = mp.get_context("spawn")
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0), "speed": np.full(54, 2.0)}
    arms, hands = Arms(), {"left": Hand(), "right": Hand()}
    session = DeviceSession(
        arms,
        hands,
        limits,
        execute=True,
        hand_control="slider",
        arm_speed_rad_s=1.0,
        qualification=SimpleNamespace(check_hand=lambda *a: None),
    )
    done = threading.Event()

    def bridge():
        try:
            with right:
                serve_connection(right, session, stopping=done.is_set, pump=lambda: None)
        finally:
            done.set()

    server = threading.Thread(target=bridge, daemon=True)
    server.start()
    now = time.monotonic()
    first = checked_chunk(
        np.zeros((50, 54)), Admission(now - 0.04, now - 0.03, now - 0.01, "first", "weights", 0), 1, limits
    )
    prepared = prepare_initial(first, np.zeros(54), limits, ROOT / "experiments/weight_motion_eval/config.yaml", now)
    incoming, status, flag = context.Queue(1), context.Queue(128), StopFlag(context)
    process = context.Process(
        target=worker, args=(left, 0, "disable", prepared, limits, {"rounds": 2}, incoming, status, flag, str(tmp_path))
    )
    process.start()
    left.close()
    events = []
    try:
        until = time.monotonic() + 20
        while time.monotonic() < until:
            try:
                event = status.get(timeout=0.2)
            except queue.Empty:
                if not process.is_alive():
                    break
                continue
            events.append(event)
            assert event["event"] != "failed", event
            if event["event"] == "infer":
                # Main-process inference sleeps while the child keeps submitting.
                sent = time.monotonic()
                count = len(arms.frames)
                time.sleep(0.12)
                assert len(arms.frames) >= count + 5
                result = checked_chunk(
                    np.full((50, 54), 0.004),
                    Admission(sent - 0.01, sent, time.monotonic(), "second", "weights", 0),
                    2,
                    limits,
                )
                incoming.put_nowait(result)
            if event["event"] == "complete":
                break
        assert any(e["event"] == "complete" for e in events), events
        process.join(5)
        assert process.exitcode == 0
        assert session.state == "released"
        assert all(not h.owns_enable for h in hands.values())
        assert arms.stopped
        assert len(arms.frames) > 250
        assert [f.sequence for f in arms.frames] == list(range(len(arms.frames)))
        assert (tmp_path / "control/events.jsonl").exists()
    finally:
        flag.set()
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join(2)
        done.set()
        server.join(2)
        for channel in (incoming, status):
            channel.cancel_join_thread()
            channel.close()


def test_disconnect_after_first_frame_stops_original_session():
    from experiments.weight_motion_eval.oneshot.core import Frame

    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    arms, hands = Arms(), {"left": Hand(), "right": Hand()}
    session = DeviceSession(
        arms,
        hands,
        {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0)},
        execute=True,
        hand_control="slider",
        qualification=SimpleNamespace(check_hand=lambda *a: None),
    )
    done = threading.Event()

    def bridge():
        with right:
            serve_connection(right, session, stopping=done.is_set, pump=lambda: None)
        done.set()

    server = threading.Thread(target=bridge, daemon=True)
    server.start()
    remote = RemoteDevices(None, execute=True, finish_policy="disable", sock=left)
    try:
        remote.call("prepare", run_id="a" * 32, plan_hash="stream", start=[0.0] * 54, finish_policy="disable")
        now = time.monotonic()
        remote.submit(Frame("a" * 32, "stream", 0, now, now + 0.02, 0.0, np.zeros(54), np.zeros(54), "playback"), now)
        remote.close()
        assert done.wait(2)
        assert session.state == "stopped"
        assert arms.stopped
        assert all(not h.owns_enable for h in hands.values())
    finally:
        remote.close()
        done.set()
        server.join(2)
