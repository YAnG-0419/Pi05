# ruff: noqa: RUF001 -- Chinese operator messages use native punctuation.
"""100 Hz process owning the existing device IPC; inference never runs here."""

import os
from pathlib import Path
import queue
import threading
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np

from deploy.fr3_wuji_slow.core import ARM
from deploy.fr3_wuji_slow.core import HAND
from deploy.fr3_wuji_slow.core import Frame
from deploy.fr3_wuji_slow.core import check_tracking
from .ipc import RemoteDevices
from deploy.fr3_wuji_slow.limits import ARM_ENDPOINT_TOLERANCE_RAD
from deploy.fr3_wuji_slow.limits import HAND_CONTACT_SETTLE_SECONDS
from deploy.fr3_wuji_slow.limits import HAND_ENDPOINT_TOLERANCE_RAD
from deploy.fr3_wuji_slow.live import LiveConsumer
from experiments.weight_motion_eval.reference import deployment_module

from .limits import ARM_TRACKING_TOLERANCE_RAD
from .smoothing import ArmSmoother
from .timeline import Timeline
from .timeline import arm_motion_settings


def settled(feedback, target):
    return (
        np.max(np.abs(feedback.positions[ARM] - target[ARM])) <= ARM_ENDPOINT_TOLERANCE_RAD
        and np.max(np.abs(feedback.velocities[ARM])) <= 0.02
        and feedback.hand_targets_reached
        and feedback.hands_settled(target, HAND_ENDPOINT_TOLERANCE_RAD)
    )


def run_control(
    devices, first, limits, options, incoming, notify, emit, stopping, *, clock=time.monotonic, sleep=time.sleep
):
    motion = arm_motion_settings(options.get("arm_motion_limits", "disabled"))
    if first.arm_motion_limits != motion["arm_motion_limits"]:
        raise ValueError("Prepared initial plan/control arm motion-limit policy mismatch")
    timeline = Timeline(first.chunk, **options)
    run_id = uuid.uuid4().hex
    player = SimpleNamespace(run_id=run_id, digest=first.digest, plan=SimpleNamespace(start=first.start))
    notify({"event": "stage", "text": "通信预热，准备双臂双手"})
    warm_until = clock() + 1.0
    while clock() < warm_until:
        if stopping():
            raise RuntimeError("Deployment cancelled before device acquisition")
        devices.feedback(clock())
        sleep(0.01)
    fb = devices.feedback(clock())

    def check_initial():
        fb.check(clock(), first.chunk.admission.epoch)
        if (
            fb.arm_speed_enforced is not motion["arm_speed_enforced"]
            or fb.arm_speed_rad_s != motion["arm_speed_rad_s"]
        ):
            raise ValueError("Fast controller/device arm motion-limit configuration mismatch")
        if fb.hand_control != "slider":
            raise ValueError("Normal-speed deployment requires the existing slider device mode")
        if not first.prepared_at <= clock() <= first.prepared_at + 30:
            raise ValueError("Initial preparation expired")
        if np.max(np.abs(fb.positions[ARM] - first.start[ARM])) > 0.005 or np.max(np.abs(fb.velocities[ARM])) > 0.02:
            raise ValueError("Initial arm pose moved before playback")

    check_initial()
    devices.prepare(player, fb, clock())
    fb = devices.feedback(clock())
    check_initial()
    phase, sequence = "approach", 0
    due = began = phase_began = clock()
    stable_since = None
    smoother = (
        ArmSmoother(first.start, limits, began, speed=motion["arm_speed_rad_s"])
        if motion["arm_speed_enforced"]
        else None
    )
    notify({"event": "arm_motion_limits", **motion})
    overall_deadline = began + first.approach.duration + options["rounds"] * 4 + 50
    notify({"event": "stage", "text": "首帧受控接近；到位后进入 30 Hz 原时间轴"})
    while not stopping():
        now = clock()
        if now < due:
            sleep(due - now)
        now = clock()
        if now - due > 0.01 or now > overall_deadline:
            raise RuntimeError("Control scheduler missed its deadline; no catch-up burst")
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
                notify({"event": "stage", "text": "最后动作块结束，等待实测到位"})
        elif phase == "approach":
            elapsed = min(now - phase_began, first.approach.duration)
            q, dq = first.approach.sample(elapsed), first.approach.sample(elapsed, 1)
            # Identical to the slow slider path: SDK owns hand rate limiting.
            q[HAND], dq[HAND] = first.chunk.actions[0, HAND], 0.0
            if elapsed >= first.approach.duration:
                phase, phase_began, stable_since = "settle_start", now, None
        else:
            q = first.chunk.actions[0].copy() if phase == "settle_start" else timeline.actions[-1].copy()
            dq = np.zeros(54)
            if (smoother is None or smoother.settled(q)) and settled(fb, q):
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
                                "arm_smoothing_max_deviation_rad": (
                                    smoother.max_deviation if smoother is not None else None
                                ),
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
                            "arm_rate_limited_values": first.chunk.arm_rate_limited_values,
                            "executed_nodes": first.chunk.actions.tolist(),
                        }
                    )
                    notify({"event": "stage", "text": "30 Hz 模型播放，100 Hz 下发，后台重推理"})
            else:
                stable_since = None
            timeout = HAND_CONTACT_SETTLE_SECONDS if fb.contact_indices(q) else 5.0
            if phase != "playback" and now - phase_began > timeout:
                raise RuntimeError("Measured endpoint settling timed out")
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise ValueError("Normal-speed command is nonfinite")
        if np.any(q < limits["lower"]) or np.any(q > limits["upper"]):
            raise ValueError("Normal-speed command outside joint limits")
        raw_arm_target = q[ARM].copy()
        if smoother is not None:
            q, dq = smoother.step(q, dq, now)
        if smoother is not None and sequence % 10 == 0:
            notify(
                {
                    "event": "arm_smoothing_sample",
                    "at": now,
                    "target": raw_arm_target.tolist(),
                    "acceleration": smoother.acceleration.tolist(),
                    "deviation_rad": float(np.max(np.abs(raw_arm_target - q[ARM]))),
                }
            )
        check_tracking(q, fb, ARM, "Normal-speed tracking", arm_tracking_rad=ARM_TRACKING_TOLERANCE_RAD)
        frame = Frame(run_id, first.digest, sequence, now, now + 0.02, now - began, q, dq, phase)
        fresh = devices.submit(frame, clock())
        if fresh is None:
            raise RuntimeError("Device bridge returned no submission feedback")
        fb = fresh
        emit(frame, fb)
        for warning in (getattr(devices, "last_submission", None) or {}).get("warnings", []):
            notify({"event": "stage", "text": "手部警告：" + warning})
        sequence += 1
        due += 0.01
        if clock() > due + 1e-9:
            raise RuntimeError("Control submission exceeded the 100 Hz tick")
    raise RuntimeError("Normal-speed execution cancelled")


def worker(sock, counter, finish_policy, first, limits, options, incoming, status, stop_flag, output):
    motion = arm_motion_settings(options.get("arm_motion_limits", "disabled"))
    devices = RemoteDevices(
        None,
        execute=True,
        finish_policy=finish_policy,
        sock=sock,
        arm_speed_enforced=motion["arm_speed_enforced"],
        arm_speed_rad_s=motion["arm_speed_rad_s"],
    )
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
