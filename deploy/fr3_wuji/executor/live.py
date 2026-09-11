"""Live shadow orchestration, using the existing read-only observation producers."""

from collections import Counter
import json
import os
from pathlib import Path
import queue
import signal
import time

import numpy as np
from observation import ObservationUnavailableError
import observe
from record_inference import latest_state

from executor.async_recorder import AsyncRecorder
from executor.inference_worker import InferenceWorker
from executor.protocol import Feedback
from executor.replay import make_limits
from executor.scheduler import Scheduler
from executor.shadow_sink import ShadowSink
from executor.state_machine import State
from executor.state_machine import StateMachine


class LiveShadow:
    def __init__(self, config, worker, *, single_step=False):
        self.worker = worker
        self.machine = StateMachine()
        self.machine.connect(time.monotonic())
        self.scheduler = Scheduler(make_limits(config), ShadowSink(self.machine.run_id), self.machine)
        self.logger = None
        self.completed = self.request_id = self.eligible = 0
        self.reasons = Counter()
        self.timings = []
        self.inflight = None
        self.clock_offset = time.time() - time.monotonic()
        self.clock_epoch = 0
        self.controls = queue.SimpleQueue()
        self.single_step = single_step

    def control(self, command):
        if command not in {"pause", "prepare"}:
            raise ValueError("Unknown shadow control")
        self.controls.put(command)

    def _feedback(self, store):
        wall, mono = time.time(), time.monotonic()
        try:
            positions, ages = latest_state(store, now_wall=wall, now_mono=mono)
        except ObservationUnavailableError:
            return None
        age = max(max(v.values()) for v in ages.values())
        return Feedback(positions, mono, mono + 0.15 - age, self.clock_epoch)

    def poll(self, store, output):
        try:
            if self.logger is None:
                self.logger = AsyncRecorder(output)
                self.logger.event(
                    {
                        "type": "started",
                        "pid": os.getpid(),
                        "run_id": self.machine.run_id,
                        "hardware_output": False,
                        "model": self.worker.metadata,
                    }
                )
            self._poll(store)
        except BaseException:
            self.scheduler.fault("live_client_failure", time.monotonic())
            raise

    def _poll(self, store):
        self.logger.check()
        now = time.monotonic()
        if abs(time.time() - now - self.clock_offset) > 0.05:
            self.clock_epoch += 1
            self.scheduler.fault("host_clock_jump", now)
            raise RuntimeError("Host clock jumped; start a new shadow session")
        while not self.controls.empty():
            command = self.controls.get_nowait()
            if command == "pause":
                self.scheduler.pause(now)
            elif self.machine.state == State.PAUSED:
                self.scheduler.resume(now)
            self.logger.event(
                {
                    "type": "control",
                    "command": command,
                    "state": self.machine.state.value,
                    "generation": self.machine.generation,
                    "mono": now,
                }
            )
        feedback = self._feedback(store)
        # Poll even when no synchronized image is available. Inference and disk
        # delays cannot suspend controls or freshness checks in this process.
        self.scheduler.tick(time.monotonic(), feedback)
        result = self.worker.poll()
        if result is None:
            return
        request, self.inflight = self.inflight, None
        if request is None or result["request_id"] != request["request_id"]:
            raise RuntimeError("Unexpected inference response identity")
        if result["kind"] == "skipped":
            self.reasons[result["reason"]] += 1
            self.logger.event({"type": "inference_skipped", **result})
            if result["reason"] == "clock_jump":
                raise RuntimeError("Clock changed in inference worker")
            return
        candidate = result["candidate"]
        feedback = self._feedback(store)
        now = time.monotonic()
        problem = self.scheduler.offer(candidate)
        decision = self.scheduler.inspect(candidate, feedback, now)
        if problem and problem not in decision["reasons"]:
            decision["reasons"].append(problem)
            decision["accepted"] = False
        self.completed += 1
        self.eligible += int(decision["accepted"])
        self.reasons.update(decision["reasons"])
        timing = {
            **result["timing"],
            "return_ipc_ms": (now - candidate.received_mono) * 1000,
            "input_age_at_evaluation_ms": (0.2 - candidate.valid_until_mono + now) * 1000,
        }
        self.timings.append(timing)
        saved = self.logger.artifact(
            candidate.request_id,
            {
                **request["observation"],
                "actions": candidate.actions,
                "state_at_evaluation": np.empty(0) if feedback is None else feedback.positions,
            },
        )
        self.logger.event(
            {
                "type": "shadow_evaluation",
                **decision,
                "timing": timing,
                "input": request["metadata"],
                "artifact_saved": saved,
                "candidate_timing": {
                    "sent_mono": candidate.sent_mono,
                    "received_mono": candidate.received_mono,
                    "valid_until_mono": candidate.valid_until_mono,
                    "clock_epoch": candidate.clock_epoch,
                },
                "feedback_timing": None
                if feedback is None
                else {
                    "sampled_mono": feedback.sampled_mono,
                    "valid_until_mono": feedback.valid_until_mono,
                    "clock_epoch": feedback.clock_epoch,
                },
            }
        )
        if self.single_step and self.machine.state == State.READY and problem is None:
            self.single_step = False
            self.machine.acquire(time.monotonic())
            feedback = self._feedback(store)
            trial = self.scheduler.tick(time.monotonic(), feedback)
            self.logger.event({"type": "single_shadow_trial", "decision": trial, "state": self.machine.state.value})
        if self.completed == 1 or self.completed % 20 == 0:
            print(
                f"Shadow {self.completed}: eligible={self.eligible}; "
                f"age={timing['input_age_at_evaluation_ms']:.1f} ms; {decision['reasons']}",
                flush=True,
            )

    def __call__(self, store, obs, metadata, output):
        if self.worker.pending or self.machine.state in {State.PAUSED, State.FAULT, State.STOPPING}:
            return
        if self._feedback(store) is None:
            return
        wall, mono = time.time(), time.monotonic()
        ages = [wall - sample["stamp"] for sample in metadata["samples"].values()]
        if min(ages) < -0.02 or max(ages) > 0.07:
            self.reasons["input_exceeds_send_budget"] += 1
            return
        if self.machine.state == State.OBSERVING:
            self.machine.ready(mono)
        request = {
            "run_id": self.machine.run_id,
            "generation": self.machine.generation,
            "request_id": self.request_id,
            "clock_epoch": self.clock_epoch,
            "capture_wall": wall,
            "capture_mono": mono,
            "valid_until_mono": mono + 0.2 - max(ages),
            "observation": obs,
            "metadata": metadata,
        }
        if self.worker.submit(request):
            self.inflight = request
            self.request_id += 1

    def finish(self, output, error=None):
        self.scheduler.pause(time.monotonic(), "client_shutdown")
        summary = {
            "mode": "live_shadow",
            "hardware_output": False,
            "status": "failed" if error else "completed",
            "error": error,
            "evaluated": self.completed,
            "eligible": self.eligible,
            "rejected": self.completed - self.eligible,
            "rejection_and_skip_reasons": dict(self.reasons),
            "shadow_frames_submitted": self.scheduler.sink.submitted,
            "inflight_discarded_at_shutdown": self.inflight is not None,
            "state": self.machine.state.value,
            "transitions": self.machine.history,
            "timing_ms": {
                name: {
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(np.max(values)),
                }
                for name in ("roundtrip_ms", "input_age_at_evaluation_ms", "return_ipc_ms")
                if (values := [item[name] for item in self.timings])
            },
        }
        try:
            if self.logger is not None:
                self.logger.event({"type": "shutdown", "state": self.machine.state.value})
        except Exception as exc:
            summary.update(status="failed", logger_error=str(exc))
            raise
        finally:
            try:
                if self.logger is not None:
                    summary["logging"] = self.logger.close()
            except Exception as exc:
                summary.update(status="failed", logger_error=str(exc))
                raise
            finally:
                if Path(output).exists():
                    (Path(output) / "client-report.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def run_live(output, config, observation_args, *, single_step=False):
    if Path(output).exists():
        raise ValueError("Use a new output directory")
    worker = InferenceWorker(config["model"]["uri"])
    try:
        consumer = LiveShadow(config, worker, single_step=single_step)
    except BaseException:
        worker.close()
        raise
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGUSR2)}
    signal.signal(signal.SIGUSR1, lambda *_: consumer.control("pause"))
    signal.signal(signal.SIGUSR2, lambda *_: consumer.control("prepare"))
    error = None
    try:
        print(f"Live shadow PID {os.getpid()}; hardware output unavailable", flush=True)
        observe.main(
            [*observation_args, "--output", str(output), "--poll-interval", "0.005", "--camera-profile", "rgb"],
            consumer=consumer,
        )
        if consumer.completed == 0:
            raise RuntimeError("No live inference completed")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        worker.close()
        consumer.finish(output, error)
