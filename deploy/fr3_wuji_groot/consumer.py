# ruff: noqa: RUF001 -- Chinese operator messages use native punctuation.
"""Fresh observation, GR00T inference and isolated control-process handoff."""

from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import uuid

import numpy as np

from experiments.weight_motion_eval.oneshot.continuous import StopFlag
from experiments.weight_motion_eval.oneshot.core import Admission

from .control import worker
from .timeline import checked_chunk
from .timeline import prepare_initial


class GrootConsumer:
    def __init__(self, devices, client, limits, output, options, config_path):
        self.devices = devices
        self.client = client
        self.model = client.metadata
        self.limits = limits
        self.output = Path(output)
        self.options = options
        self.config_path = config_path
        self.context = mp.get_context("spawn")
        self.incoming, self.status = self.context.Queue(1), self.context.Queue(128)
        self.stop_flag = StopFlag(self.context)
        self.process = None
        self.closed = False
        self.completed = 0
        self.request = {"number": 1}
        self.report = {
            "state": "waiting_for_observation",
            "backend": "groot_n1d7",
            "source_hz": 30,
            "control_hz": 100,
            "action_horizon": 16,
            "hardware_output": True,
            "options": options,
            "model": self.model,
            "chunks": [],
        }

    def save_report(self):
        (self.output / "groot-report.json").write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n")

    def poll(self):
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
                    raise RuntimeError("Multiple outstanding GR00T inference requests")
                self.request = event
            elif kind == "stage":
                print("【阶段】" + event["text"], flush=True)
            elif kind == "chunk_started":
                self.report["chunks"].append({key: value for key, value in event.items() if key != "executed_nodes"})
                print(f"【阶段】GR00T 动作块 {event['number']}/{self.options['rounds']}", flush=True)
            elif kind == "complete":
                self.completed = 1
                self.report.update(state="complete", result=event)
                self.save_report()
        if (
            self.process is not None
            and not self.process.is_alive()
            and (not self.completed or self.process.exitcode != 0 or self.devices.finish_policy == "hold")
        ):
            raise RuntimeError("GR00T control process exited unexpectedly or hold monitoring was lost")

    def __call__(self, store, observation, metadata, directory):
        try:
            self.consume(observation, metadata)
        except BaseException:
            self.stop_flag.set()
            raise

    def consume(self, observation, metadata):
        self.poll()
        if self.completed or self.request is None:
            return
        sent, wall = time.monotonic(), time.time()
        ages = [wall - sample["stamp"] for sample in metadata["samples"].values()]
        if not ages or any(not 0 <= age <= 0.07 for age in ages):
            return
        number = self.request["number"]
        self.request = None
        print(f"【推理】GR00T 第 {number} 次，使用新观测", flush=True)
        remaining = max(0.001, 1.0 - max(ages) - (time.monotonic() - sent))
        response = self.client.infer(observation, timeout=remaining)
        received = time.monotonic()
        if abs(time.time() - wall - (received - sent)) > 0.05:
            raise ValueError("Host clock changed during GR00T inference")
        admission = Admission(
            sent - max(ages),
            sent,
            received,
            uuid.uuid4().hex,
            self.model["model_manifest_sha256"],
            0,
        )
        admission.check()
        raw = np.asarray(response["actions"])
        directory = self.output / f"chunk-{number:04d}"
        directory.mkdir()
        np.savez_compressed(directory / "inference.npz", actions=raw, state=observation["observation/state"])
        (directory / "admission.json").write_text(
            json.dumps(
                {
                    **asdict(admission),
                    "model_instruction": self.model["instruction"],
                    "observation_prompt_ignored": observation.get("prompt"),
                    "observation_metadata": metadata,
                    "policy_timing": response.get("policy_timing"),
                },
                indent=2,
            )
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
        for channel in (self.incoming, self.status):
            channel.cancel_join_thread()
            channel.close()
