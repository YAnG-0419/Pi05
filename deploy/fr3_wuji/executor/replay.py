"""Replay recorded timing on a virtual clock; never connects to model or hardware."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from record_inference import load_limits

from executor.async_recorder import AsyncRecorder
from executor.protocol import ARM_INDICES
from executor.protocol import Candidate
from executor.protocol import Feedback
from executor.scheduler import Limits
from executor.scheduler import Scheduler
from executor.shadow_sink import ShadowSink
from executor.state_machine import StateMachine


def make_limits(config):
    reference = load_limits()
    if reference["arm_initial_delta"] != config["arm"]["initial_delta_rad"] or np.any(
        reference["speed"][ARM_INDICES] != config["arm"]["max_joint_speed_rad_s"]
    ):
        raise ValueError("Reference arm configuration changed; review before shadow evaluation")
    return Limits(
        reference["lower"],
        reference["upper"],
        reference["speed"],
        arm_initial_delta=config["arm"]["initial_delta_rad"],
        hand_initial_delta=config["hand"].get("initial_delta_rad"),
    )


def run_replay(source, output, config, *, inject=None):
    source, output = Path(source), Path(output)
    records = [json.loads(line) for line in (source / "inferences.jsonl").read_text().splitlines()]
    if not records:
        raise ValueError("No inference records")
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    logger = AsyncRecorder(output)
    machine = StateMachine()
    machine.connect(0)
    machine.ready(0)
    scheduler = Scheduler(make_limits(config), ShadowSink(machine.run_id), machine)
    reasons, total, eligible = Counter(), 0, 0
    origin = records[0]["send_wall"]
    summary = {
        "mode": "replay_shadow",
        "hardware_output": False,
        "source": str(source.resolve()),
        "virtual_clock": True,
        "injection": inject,
        "status": "running",
    }
    try:
        for record in records:
            index = record["index"]
            with np.load(source / f"inference-{index:04d}.npz", allow_pickle=False) as stored:
                send = record["send_wall"] - origin
                receive = send + record["roundtrip_seconds"]
                deadline = send + 0.2 - max(record["input_age_at_send_seconds"].values())
                candidate = Candidate(
                    machine.run_id,
                    machine.generation,
                    index,
                    str(record["input"]["target_stamp"]),
                    send,
                    receive,
                    deadline,
                    stored["actions"],
                    stored["observation/state"],
                )
                feedback = None
                if record["latest_state_at_return"] is not None and stored["state_at_return"].shape == (54,):
                    age = max(
                        max(v["source_age_seconds"], v["receive_age_seconds"])
                        for v in record["latest_state_at_return"].values()
                    )
                    feedback = Feedback(stored["state_at_return"], receive, receive + 0.15 - age)
                now = receive
                if inject == "late_response":
                    now = max(receive, deadline + 0.001)
                elif inject == "feedback_loss":
                    feedback = None
                elif inject == "pause_before_response" and total == 0:
                    scheduler.pause(send, "injected_pause")
                problem = scheduler.offer(candidate)
                decision = scheduler.inspect(candidate, feedback, now)
                if record.get("clock_jump_during_inference"):
                    decision["reasons"].append("recorded_clock_jump")
                if max(record["input_age_at_send_seconds"].values()) > 0.07:
                    decision["reasons"].append("input_exceeds_send_budget")
                if problem and problem not in decision["reasons"]:
                    decision["reasons"].append(problem)
                decision["accepted"] = not decision["reasons"]
                logger.event({"type": "shadow_evaluation", **decision, "clock": "recorded_relative"})
                total += 1
                eligible += int(decision["accepted"])
                reasons.update(decision["reasons"])
        summary.update(
            status="completed",
            evaluated=total,
            eligible=eligible,
            rejected=total - eligible,
            rejection_reasons=dict(reasons),
            shadow_frames_submitted=scheduler.sink.submitted,
            state=machine.state.value,
            transitions=machine.history,
            note="Every recorded chunk is inspected as a hypothetical initial target. No automatic trials; "
            "unknown hand threshold is reported, not guessed. Playback clocks cannot drive hardware.",
        )
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            summary["logging"] = logger.close()
        except Exception as error:
            summary.update(status="failed", logger_error=str(error))
            raise
        finally:
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
