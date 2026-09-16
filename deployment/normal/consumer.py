# ruff: noqa: RUF001 -- Chinese operator messages use native punctuation.
"""Observation callback and WebSocket inference in the non-control process."""

from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import uuid

import numpy as np

from experiments.weight_motion_eval.oneshot.continuous import StopFlag
from experiments.weight_motion_eval.oneshot.core import Admission

from .control import worker
from .timeline import checked_chunk
from .timeline import prepare_initial


class FastConsumer:
    def __init__(self, devices, ws, model, limits, output, options, config_path):
        from openpi_client import msgpack_numpy

        self.codec, self.devices, self.ws = msgpack_numpy, devices, ws
        self.model, self.limits, self.output = model, limits, Path(output)
        self.options, self.config_path = options, config_path
        self.context = mp.get_context("spawn")
        self.incoming, self.status = self.context.Queue(1), self.context.Queue(128)
        self.stop_flag = StopFlag(self.context)
        self.process = None
        self.closed = False
        self.completed = 0
        self.request = {"number": 1}
        self.report = {
            "state": "waiting_for_observation",
            "source_hz": 30,
            "control_hz": 100,
            "hardware_output": True,
            "options": options,
            "model": model,
            "chunks": [],
        }

    def save_report(self):
        (self.output / "fast-report.json").write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n")

    def poll(self, store=None, output=None):
        while True:
            try:
                event = self.status.get_nowait()
            except queue.Empty:
                break
            kind = event["event"]
            if kind == "failed":
                self.report.update(state="failed", error=event["error"], traceback=event.get("traceback"))
                self.save_report()
                raise RuntimeError(event["error"])
            if kind == "infer":
                if self.request is not None:
                    raise RuntimeError("Multiple outstanding inference requests")
                self.request = event
            elif kind == "stage":
                print("【阶段】" + event["text"], flush=True)
            elif kind == "chunk_started":
                self.report["chunks"].append({k: v for k, v in event.items() if k != "executed_nodes"})
                limited = event.get("arm_rate_limited_values", 0)
                print(
                    f"【阶段】动作块 {event['number']}/{self.options['rounds']}："
                    f"30 Hz，延迟补偿跳过 {event['latency_offset_steps']} 步"
                    + (f"，机械臂 {limited} 个目标按 1 rad/s 限幅" if limited else ""),
                    flush=True,
                )
            elif kind == "complete":
                self.completed = 1
                self.report.update(state="complete", result=event)
                self.save_report()
        if (
            self.process is not None
            and not self.process.is_alive()
            and (not self.completed or self.process.exitcode != 0 or self.devices.finish_policy == "hold")
        ):
            raise RuntimeError("Control process exited unexpectedly or hold monitoring was lost")

    def __call__(self, store, obs, metadata, directory):
        try:
            self.consume(store, obs, metadata, directory)
        except BaseException as error:
            # observe.main cleans up cameras before returning. Stop motion now,
            # then expose the original failure before its potentially slow cleanup.
            self.stop_flag.set()
            detail = f"{type(error).__name__}: {error}"
            self.report.update(state="failed", error=detail)
            print("【执行失败】" + detail, file=sys.stderr, flush=True)
            try:
                self.save_report()
            except OSError as report_error:
                print("【报告写入失败】" + str(report_error), file=sys.stderr, flush=True)
            raise

    def consume(self, store, obs, metadata, directory):
        self.poll()
        if self.completed or self.request is None:
            return
        packet = self.codec.packb(obs)
        sent, wall = time.monotonic(), time.time()
        ages = [wall - sample["stamp"] for sample in metadata["samples"].values()]
        if not ages or any(not 0 <= age <= 0.07 for age in ages):
            return
        number = self.request["number"]
        self.request = None
        print(f"【推理】第 {number} 次，使用新观测", flush=True)
        self.ws.send(packet)
        response = self.ws.recv(timeout=max(0.001, 1.0 - max(ages) - (time.monotonic() - sent)))
        received = time.monotonic()
        if abs(time.time() - wall - (received - sent)) > 0.05:
            raise ValueError("Host clock changed during inference")
        if isinstance(response, str):
            raise RuntimeError("Policy server returned an error: " + response)
        admission = Admission(
            sent - max(ages), sent, received, uuid.uuid4().hex, self.model["checkpoint_manifest_sha256"], 0
        )
        admission.check()
        raw = np.asarray(self.codec.unpackb(response)["actions"])
        # Persist every original prediction, including rejected ones.
        directory = self.output / f"chunk-{number:04d}"
        directory.mkdir()
        np.savez_compressed(directory / "inference.npz", actions=raw, state=obs["observation/state"])
        (directory / "admission.json").write_text(
            json.dumps({**asdict(admission), "prompt": obs.get("prompt"), "observation_metadata": metadata}, indent=2)
            + "\n"
        )
        chunk = checked_chunk(raw, admission, number, self.limits)
        if chunk.projected_hand_values:
            print(f"【手部预测】{chunk.projected_hand_values} 个角度投影到现有限位；原始预测已保存", flush=True)
        self.poll()
        if self.process is None:
            feedback = self.devices.feedback(time.monotonic())
            initial = prepare_initial(chunk, feedback.positions, self.limits, self.config_path, time.monotonic())
            self.report.update(
                state="running", stream_digest=initial.digest, approach_seconds=initial.approach.duration
            )
            self.process = self.context.Process(
                target=worker,
                args=(
                    self.devices.sock,
                    self.devices.counter,
                    self.devices.finish_policy,
                    initial,
                    self.limits,
                    self.options,
                    self.incoming,
                    self.status,
                    self.stop_flag,
                    str(self.output),
                ),
            )
            self.process.start()
            # Exactly one IPC owner after transfer; parent never sends motion.
            self.devices.sock.close()
            self.devices.closed = True
        else:
            self.incoming.put_nowait(chunk)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_flag.set()
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        if self.report["state"] not in {"complete", "failed"}:
            self.report["state"] = "stopped"
        self.save_report()
        for pipe in (self.incoming, self.status):
            pipe.cancel_join_thread()
            pipe.close()
