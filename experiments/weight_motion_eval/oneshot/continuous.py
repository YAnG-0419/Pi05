# ruff: noqa: RUF001 -- Chinese terminal messages use Chinese punctuation.
"""Finite receding-horizon execution with an independent motion process.

Model inference, spline planning and artifact writes stay in the parent. The
motion process keeps the same gateway session/sequence across all chunks.
"""

from dataclasses import asdict
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np

from .core import Admission
from .core import OneShot
from .ipc import RemoteDevices
from .ipc import wire_feedback
from .live import LiveConsumer


class StopFlag:
    """Shared cancellation byte; no condition lock can be stranded on child exit."""

    def __init__(self, context):
        self.value = context.RawValue("b", 0)

    def set(self):
        self.value.value = 1

    def is_set(self):
        return bool(self.value.value)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while not self.is_set() and time.monotonic() < end:
            time.sleep(min(0.01, max(0.0, end - time.monotonic())))
        return self.is_set()


def control_loop(
    devices,
    first,
    incoming,
    notify,
    emit,
    stopping,
    rounds,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    idle_timeout=10.0,
):
    player, sequence, round_number = first, 0, 1
    notify({"event": "stage", "stage": "通信预热：连续检查四设备反馈 1 秒"})
    warm_until = clock() + 1.0
    while clock() < warm_until:
        if stopping():
            raise RuntimeError("Feedback warmup cancelled")
        devices.feedback(clock())
        sleep(0.01)
    fb = devices.feedback(clock())
    player.check_start(fb, clock())
    notify({"event": "stage", "stage": "设备准备：确认手部使能并刷新两臂反馈"})
    devices.prepare(player, fb, clock())
    fb = devices.feedback(clock())
    player.start(fb, clock())
    notify({"event": "stage", "stage": "第 1 轮归位：慢速移到模型第一帧"})
    due, last, waiting_since, last_status = clock(), None, None, 0.0
    while not stopping():
        now = clock()
        if now < due:
            sleep(due - now)
        now = clock()
        if now - due > 0.01:
            raise RuntimeError(f"Control wake deadline missed: lateness_ms={(now - due) * 1000:.3f}")
        # Reuse the measured sample from the preceding device submit. Its source
        # timestamps are preserved and checked, never refreshed by the client.
        fb.check(now, fb.epoch)
        if waiting_since is not None:
            if now - waiting_since > idle_timeout:
                raise RuntimeError("Next fresh plan did not arrive within 10 seconds")
            try:
                candidate = incoming.get_nowait()
            except queue.Empty:
                candidate = None
            if candidate is not None:
                candidate.run_id = player.run_id
                candidate.check_start(fb, now)
                result = devices.call(
                    "next_plan", run_id=player.run_id, plan_hash=candidate.digest, start=candidate.plan.start.tolist()
                )
                if result["sequence"] != sequence:
                    raise RuntimeError("Next-plan sequence mismatch")
                player = candidate
                player.sequence = sequence
                player.start(fb, clock())
                waiting_since = None
                round_number += 1
                notify({"event": "round_started", "round": round_number})
                notify({"event": "stage", "stage": f"第 {round_number} 轮：平滑衔接新动作段"})
                now = clock()
        if waiting_since is None:
            previous_phase = player.state
            frame = player.tick(fb, now)
            if player.state != previous_phase and player.state not in {"fault", "stopping", "complete"}:
                descriptions = {
                    "settle_start": "等待起始姿态到位",
                    "playback": "慢速播放动作段",
                    "settle_end": "等待本轮终点到位",
                }
                notify(
                    {"event": "stage", "stage": f"第 {round_number} 轮：{descriptions.get(player.state, player.state)}"}
                )
            if player.state == "complete":
                notify(
                    {
                        "event": "round_complete",
                        "round": round_number,
                        "transitions": player.transitions,
                        "target": last.positions.tolist(),
                    }
                )
                if round_number == rounds:
                    devices.finish()
                    notify({"event": "complete", "rounds": rounds})
                    return
                waiting_since = now
                notify({"event": "stage", "stage": "轮间保持：采集新观测并等待下一轮推理"})
                frame = None
            elif player.state in {"fault", "stopping"}:
                raise RuntimeError(f"Round {round_number}: {player.reason}")
        else:
            frame = None
        if frame is None:
            # Hold frames share the active plan identity and global wire sequence.
            frame = replace(
                last,
                sequence=sequence,
                created=now,
                valid_until=now + 0.02,
                velocities=np.zeros(54),
                phase="settle_end",
            )
        submitted = devices.submit(frame, clock())
        for warning in getattr(devices, "last_submission", {}).get("warnings", []):
            notify({"event": "stage", "stage": "【手部警告】" + warning})
        if submitted is None:
            raise RuntimeError("Device bridge lacks submission feedback support")
        fb, last, sequence = submitted, frame, frame.sequence + 1
        emit(frame, fb)
        if waiting_since is not None and now - last_status >= 0.05:
            notify(
                {
                    "event": "between",
                    "round": round_number,
                    "target": last.positions.tolist(),
                    "feedback": wire_feedback(fb),
                }
            )
            last_status = now
        due += 0.01
        if clock() > due:
            raise RuntimeError(f"Control submission exceeded 100 Hz tick: overrun_ms={(clock() - due) * 1000:.3f}")
    raise RuntimeError("Continuous execution cancelled")


def _worker(sock, counter, finish_policy, first, incoming, status, stop_event, output, rounds):
    devices = RemoteDevices(None, execute=True, finish_policy=finish_policy, sock=sock)
    devices.counter = counter
    recorder = None
    local = queue.Queue(maxsize=1)
    parent_pid = os.getppid()

    def watch_parent():
        while not stop_event.wait(0.05):
            if os.getppid() != parent_pid:
                stop_event.set()
                return

    def load_plans():
        # Unpickle spline data outside the control thread.
        while not stop_event.is_set():
            try:
                item = incoming.get(timeout=0.1)
                local.put(item, timeout=0.1)
            except queue.Empty:
                continue
            except BaseException:
                stop_event.set()
                return

    def notify(event):
        status.put_nowait(event)

    try:
        from experiments.weight_motion_eval.reference import deployment_module

        recorder = deployment_module("executor.async_recorder").AsyncRecorder(Path(output) / "control")
        emitter = LiveConsumer.__new__(LiveConsumer)
        emitter.devices, emitter.recorder = devices, recorder
        threading.Thread(target=load_plans, daemon=True).start()
        threading.Thread(target=watch_parent, daemon=True).start()
        control_loop(devices, first, local, notify, emitter.emit, stop_event.is_set, rounds)
        while finish_policy == "hold" and not stop_event.wait(0.05):
            if devices.hold_error:
                raise RuntimeError(devices.hold_error)
    except BaseException as error:
        # Stop before traceback formatting, reporting, or draining the recorder.
        try:
            devices.stop()
        finally:
            notify(
                {"event": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            )
    finally:
        try:
            devices.stop()
        finally:
            devices.close()
            if recorder is not None:
                recorder.close()


class ContinuousConsumer(LiveConsumer):
    def __init__(self, *args, rounds=50, replan_steps=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = dict(self.config, execution_steps=replan_steps)
        self.config["project_hand_predictions"] = self.config.get("hand_control") == "slider"
        self.rounds, self.rounds_completed = rounds, 0
        self.process = None
        self.between = None
        self.pending = False
        self.context = mp.get_context("spawn")
        self.incoming = self.context.Queue(1)
        self.status = self.context.Queue(256)
        self.stop_event = StopFlag(self.context)
        self.report.update(mode="continuous", rounds=rounds, replan_steps=replan_steps)

    def poll(self, store=None, output=None):
        while True:
            try:
                event = self.status.get_nowait()
            except queue.Empty:
                break
            kind = event["event"]
            if kind == "stage":
                self.report["stage"] = event["stage"]
                print("【阶段】" + event["stage"], flush=True)
                continue
            if kind == "failed":
                print("【阶段】执行异常，停止设备：" + event["error"], flush=True)
                self.report.update(state="failed", error=event["error"], traceback=event.get("traceback"))
                self.save_report()
                raise RuntimeError(event["error"])
            if kind == "round_complete":
                self.rounds_completed = event["round"]
                self.pending = False
                self.recorder.event(event)
                print(f"【阶段】第 {self.rounds_completed}/{self.rounds} 轮已完成", flush=True)
            elif kind == "between" and event["round"] == self.rounds_completed:
                self.between = event
            elif kind == "round_started":
                self.pending = False
                self.between = None
            elif kind == "complete":
                print("【阶段】全部推理轮次完成，执行结束策略", flush=True)
                self.completed = 1
                self.report.update(state="complete", rounds_completed=self.rounds_completed)
                self.save_report()
        if self.process is not None and not self.process.is_alive() and not self.completed:
            raise RuntimeError("Continuous control process exited before completion")

    def save_report(self):
        (self.output / "continuous-report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def __call__(self, store, obs, metadata, directory):
        try:
            return self.consume(store, obs, metadata, directory)
        except BaseException as error:
            self.report.update(state="failed", error=f"{type(error).__name__}: {error}")
            self.close()
            raise

    def consume(self, store, obs, metadata, directory):
        self.poll()
        if self.completed or self.pending or (self.process is not None and self.between is None):
            return
        packet = self.codec.packb(obs)
        sent, wall = time.monotonic(), time.time()
        ages = [wall - value["stamp"] for value in metadata["samples"].values()]
        if not ages or any(not 0 <= age <= 0.07 for age in ages):
            return
        print(f"【阶段】第 {self.rounds_completed + 1}/{self.rounds} 轮：使用实时观测进行模型推理", flush=True)
        self.requested = True
        self.ws.send(packet)
        result = self.ws.recv(timeout=1.0 - max(ages))
        received = time.monotonic()
        if abs((time.time() - wall) - (received - sent)) > 0.05:
            raise ValueError("Host clock changed during inference")
        if isinstance(result, str):
            raise RuntimeError("Policy server returned an error: " + result)
        import uuid

        admission = Admission(
            sent - max(ages), sent, received, uuid.uuid4().hex, self.model["checkpoint_manifest_sha256"], 0
        )
        admission.check()
        actions = np.asarray(self.codec.unpackb(result)["actions"])
        self.poll()
        from experiments.weight_motion_eval.planner import build_plan
        from experiments.weight_motion_eval.planner import save_plan

        if self.process is None:
            start = self.devices.feedback(time.monotonic()).positions
        else:
            start = np.asarray(self.between["target"])
        config = dict(self.config, continuation=self.process is not None, inference_request_id=admission.request_id)
        print("【阶段】推理返回，规划慢速动作段", flush=True)
        round_dir = self.output / f"round-{self.rounds_completed + 1:04d}"
        round_dir.mkdir()
        # Preserve the full prediction even when planning rejects its prefix.
        np.savez_compressed(round_dir / "inference.npz", actions=actions,
                            state=obs["observation/state"], recorded_start=start)
        (round_dir / "admission.json").write_text(json.dumps(asdict(admission), indent=2) + "\n")
        try:
            plan = build_plan(actions, start, config, self.limits, self.names)
            player = OneShot(plan, admission, time.monotonic())
        except ValueError as error:
            (round_dir / "planning-error.json").write_text(
                json.dumps({"error": str(error), "execution_steps": config.get("execution_steps", 50)}, indent=2) + "\n"
            )
            raise
        save_plan(plan, round_dir / "plan")
        projection = plan.report["hand_prediction_projection"]
        if projection["adjustment_count"]:
            print(f"【阶段】【手部预测警告】{projection['adjustment_count']} 个预测角度超出关节范围，"
                  f"已限制到合法边界；最大调整 {projection['max_adjustment_rad']:.4f} rad，继续执行", flush=True)
            self.recorder.event({"event": "hand_prediction_projected",
                                 "round": self.rounds_completed + 1, **projection})
        self.recorder.event(
            {
                "event": "round_admitted",
                "round": self.rounds_completed + 1,
                "plan_hash": player.digest,
                "admission": asdict(admission),
            }
        )
        if self.process is None:
            self.process = self.context.Process(
                target=_worker,
                args=(
                    self.devices.sock,
                    self.devices.counter,
                    self.devices.finish_policy,
                    player,
                    self.incoming,
                    self.status,
                    self.stop_event,
                    str(self.output),
                    self.rounds,
                ),
            )
            self.process.start()
            # Transfer the sole IPC connection to the motion process.
            self.devices.sock.close()
            self.devices.closed = True
        else:
            self.incoming.put_nowait(player)
        self.pending = True
        print(f"【阶段】第 {self.rounds_completed + 1} 轮计划就绪：预测 50 步，执行前 {len(plan.raw)} 步", flush=True)

    def close(self):
        self.stop_event.set()
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        if self.report.get("state") not in {"complete", "failed"}:
            self.report["state"] = "stopped"
        self.save_report()
