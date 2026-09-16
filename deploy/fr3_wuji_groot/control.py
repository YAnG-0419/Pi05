# ruff: noqa: RUF001 -- Chinese operator messages use native punctuation.
"""100 Hz hardware process for serial sixteen-step GR00T chunks."""

import os
from pathlib import Path
import queue
import threading
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np

from deploy.fr3_wuji_fast.control import settled
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import HAND
from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.core import check_tracking
from experiments.weight_motion_eval.oneshot.ipc import RemoteDevices
from experiments.weight_motion_eval.oneshot.limits import HAND_CONTACT_SETTLE_SECONDS
from experiments.weight_motion_eval.oneshot.live import LiveConsumer
from experiments.weight_motion_eval.reference import deployment_module

from .timeline import Timeline


def run_control(
    devices, first, limits, options, incoming, notify, emit, stopping, *, clock=time.monotonic, sleep=time.sleep
):
    timeline = Timeline(first.chunk, **options)
    run_id = uuid.uuid4().hex
    player = SimpleNamespace(run_id=run_id, digest=first.digest, plan=SimpleNamespace(start=first.start))
    notify({"event": "stage", "text": "通信预热，准备双臂双手"})
    warm_until = clock() + 1.0
    while clock() < warm_until:
        if stopping():
            raise RuntimeError("GR00T deployment cancelled before device acquisition")
        devices.feedback(clock())
        sleep(0.01)
    fb = devices.feedback(clock())

    def check_initial():
        fb.check(clock(), first.chunk.admission.epoch)
        if fb.hand_control != "slider":
            raise ValueError("GR00T deployment requires the existing slider device mode")
        if not first.prepared_at <= clock() <= first.prepared_at + 30:
            raise ValueError("Initial GR00T preparation expired")
        if np.max(np.abs(fb.positions[ARM] - first.start[ARM])) > 0.005 or np.max(np.abs(fb.velocities[ARM])) > 0.02:
            raise ValueError("Initial arm pose moved before GR00T playback")

    check_initial()
    devices.prepare(player, fb, clock())
    fb = devices.feedback(clock())
    check_initial()
    phase, sequence = "approach", 0
    due = began = phase_began = clock()
    stable_since = None
    overall_deadline = began + first.approach.duration + options["rounds"] * 4 + 50
    notify({"event": "stage", "text": "首帧受控接近；到位后进入 GR00T 30 Hz 串行动作块"})
    while not stopping():
        now = clock()
        if now < due:
            sleep(due - now)
        now = clock()
        if now - due > 0.01 or now > overall_deadline:
            raise RuntimeError("GR00T control scheduler missed its deadline; no catch-up burst")
        fb.check(now, first.chunk.admission.epoch)
        if phase == "playback":
            timeline.check_pending(now)
            try:
                chunk = incoming.get_nowait()
            except queue.Empty:
                chunk = None
            if chunk is not None:
                notify(timeline.install(chunk, now, limits))
            if timeline.request_due(now):
                notify(timeline.request(now))
            q, dq = timeline.sample(now)
            if timeline.number == timeline.rounds and timeline.exhausted(now):
                phase, phase_began, stable_since = "settle_end", now, None
                notify({"event": "stage", "text": "最后 GR00T 动作块结束，等待实测到位"})
        elif phase == "approach":
            elapsed = min(now - phase_began, first.approach.duration)
            q, dq = first.approach.sample(elapsed), first.approach.sample(elapsed, 1)
            q[HAND], dq[HAND] = first.chunk.actions[0, HAND], 0.0
            if elapsed >= first.approach.duration:
                phase, phase_began, stable_since = "settle_start", now, None
        else:
            q = first.chunk.actions[0].copy() if phase == "settle_start" else timeline.actions[-1].copy()
            dq = np.zeros(54)
            if settled(fb, q):
                stable_since = now if stable_since is None else stable_since
                if now - stable_since >= 0.5:
                    if phase == "settle_end":
                        devices.finish()
                        notify(
                            {
                                "event": "complete",
                                "chunks": timeline.number,
                                "elapsed_seconds": now - began,
                                "frames": sequence,
                            }
                        )
                        return
                    phase, phase_began = "playback", now
                    timeline.start(now)
                    notify(
                        {
                            "event": "chunk_started",
                            "number": 1,
                            "at": now,
                            "latency_offset_steps": 0,
                            "sha256": first.chunk.sha256,
                            "projected_hand_values": first.chunk.projected_hand_values,
                            "executed_nodes": first.chunk.actions.tolist(),
                        }
                    )
                    notify({"event": "stage", "text": "GR00T 16 步动作按 30 Hz 播放，100 Hz 下发"})
            else:
                stable_since = None
            timeout = HAND_CONTACT_SETTLE_SECONDS if fb.contact_indices(q) else 5.0
            if phase != "playback" and now - phase_began > timeout:
                raise RuntimeError("Measured endpoint settling timed out")
        check_tracking(q, fb, ARM, "GR00T tracking")
        frame = Frame(run_id, first.digest, sequence, now, now + 0.02, now - began, q, dq, phase)
        fresh = devices.submit(frame, clock())
        if fresh is None:
            raise RuntimeError("Device bridge returned no GR00T submission feedback")
        fb = fresh
        emit(frame, fb)
        for warning in (getattr(devices, "last_submission", None) or {}).get("warnings", []):
            notify({"event": "stage", "text": "手部警告：" + warning})
        sequence += 1
        due += 0.01
        if clock() > due + 1e-9:
            raise RuntimeError("GR00T control submission exceeded the 100 Hz tick")
    raise RuntimeError("GR00T execution cancelled")


def worker(sock, counter, finish_policy, first, limits, options, incoming, status, stop_flag, output):
    devices = RemoteDevices(None, execute=True, finish_policy=finish_policy, sock=sock)
    devices.counter = counter
    recorder = None
    local = queue.Queue(maxsize=1)
    parent = os.getppid()

    def watch_parent():
        while not stop_flag.wait(0.05):
            if os.getppid() != parent:
                stop_flag.set()
                return

    def load_chunks():
        while not stop_flag.is_set():
            try:
                local.put(incoming.get(timeout=0.1), timeout=0.1)
            except queue.Empty:
                continue
            except BaseException:
                stop_flag.set()
                return

    def notify(event):
        status.put_nowait(event)
        if recorder is not None:
            recorder.event(event)

    try:
        recorder = deployment_module("executor.async_recorder").AsyncRecorder(Path(output) / "control")
        emitter = LiveConsumer.__new__(LiveConsumer)
        emitter.devices, emitter.recorder = devices, recorder
        threading.Thread(target=watch_parent, daemon=True).start()
        threading.Thread(target=load_chunks, daemon=True).start()
        run_control(devices, first, limits, options, local, notify, emitter.emit, stop_flag.is_set)
        while finish_policy == "hold" and not stop_flag.wait(0.05):
            if devices.hold_error:
                raise RuntimeError(devices.hold_error)
    except BaseException as error:
        try:
            devices.stop()
        finally:
            status.put_nowait(
                {"event": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            )
    finally:
        try:
            devices.stop()
        finally:
            devices.close()
            if recorder is not None:
                recorder.close()
