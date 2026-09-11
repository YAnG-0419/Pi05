"""One fresh inference to one finite trajectory through the device IPC bridge."""

from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .core import Admission
from .core import OneShot
from .runner import run_live


class LiveConsumer:
    def __init__(self, devices, ws, model, config, limits, names, output, recorder):
        from openpi_client import msgpack_numpy

        self.codec = msgpack_numpy
        self.devices, self.ws, self.model = devices, ws, model
        self.config, self.limits, self.names, self.output = config, limits, names, Path(output)
        self.recorder, self.completed = recorder, 0
        self.requested = False
        self.player = None
        self.report = {"hardware_output": True, "model": model, "state": "waiting_for_fresh_observation"}

    def __call__(self, store, obs, metadata, directory):
        if self.requested:
            return
        packet = self.codec.packb(obs)
        sent, wall = time.monotonic(), time.time()
        ages = [wall - sample["stamp"] for sample in metadata["samples"].values()]
        if not ages or any(not 0 <= age <= 0.07 for age in ages):
            return  # Select another observation before issuing the only request.
        self.devices.feedback(sent)
        # Device checks may take time; admission uses the final send instant.
        sent, wall = time.monotonic(), time.time()
        ages = [wall - sample["stamp"] for sample in metadata["samples"].values()]
        if any(not 0 <= age <= 0.07 for age in ages):
            return
        self.requested = True
        self.ws.send(packet)
        result = self.ws.recv(timeout=0.2 - max(ages))
        received = time.monotonic()
        if abs((time.time() - wall) - (received - sent)) > 0.05:
            raise ValueError("Host clock changed during inference")
        if isinstance(result, str):
            raise RuntimeError("Policy server returned an error: " + result)
        admission = Admission(
            sent - max(ages), sent, received, uuid.uuid4().hex, self.model["checkpoint_manifest_sha256"], 0
        )
        admission.check()  # Before planning, serialization or device acquisition.
        actions = np.asarray(self.codec.unpackb(result)["actions"])
        feedback = self.devices.feedback(time.monotonic())
        from experiments.weight_motion_eval.planner import build_plan
        from experiments.weight_motion_eval.planner import save_plan

        plan = build_plan(actions, feedback.positions, self.config, self.limits, self.names)
        self.player = OneShot(plan, admission, time.monotonic())
        save_plan(plan, self.output / "plan")
        np.savez_compressed(
            self.output / "live-inference.npz",
            **{k: v for k, v in obs.items() if k != "prompt"},
            prompt=np.array(obs["prompt"]),
            actions=actions,
            state_at_return=feedback.positions,
        )
        self.report.update(
            state="prepared",
            admission=asdict(admission),
            observation_metadata=metadata,
            plan_hash=self.player.digest,
            run_id=self.player.run_id,
        )
        self.recorder.event({"event": "inference_admitted", **self.report})
        try:
            run_live(self.player, self.devices, stop_requested=lambda: False, emit=self.emit)
            self.report.update(state=self.player.state, reason=self.player.reason, transitions=self.player.transitions)
            if self.player.state != "complete":
                raise RuntimeError("One-shot motion did not complete: " + str(self.player.reason))
            self.completed = 1
            self.recorder.event({"event": "trajectory_complete", "run_id": self.player.run_id})
        except BaseException as error:
            self.report.update(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            (self.output / "live-report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def emit(self, frame, feedback):
        submission = getattr(self.devices, "last_submission", None) or {}
        self.recorder.event(
            {
                "event": "frame_submitted",
                "sequence": frame.sequence,
                "created": frame.created,
                "valid_until": frame.valid_until,
                "phase": frame.phase,
                "command": frame.positions.tolist(),
                "hand_output": submission.get("hands"),
                "measured": feedback.positions.tolist(),
                "measured_velocity": feedback.velocities.tolist(),
            }
        )
