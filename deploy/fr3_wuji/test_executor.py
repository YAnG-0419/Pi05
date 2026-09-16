"""Execution-core invariants and failure paths; no hardware or model server."""

import dataclasses
from pathlib import Path
import time

from executor.async_recorder import AsyncRecorder
from executor.config import load_config
from executor.protocol import ARM_INDICES
from executor.protocol import ARM_NAMES
from executor.protocol import Candidate
from executor.protocol import Feedback
from executor.protocol import OutputFrame
from executor.protocol import arm_payload
from executor.scheduler import Limits
from executor.scheduler import Scheduler
from executor.shadow_sink import ShadowSink
from executor.state_machine import State
from executor.state_machine import StateMachine
import numpy as np
import pytest


def setup(*, hand_limit=0.05):
    limits = Limits(np.full(54, -2), np.full(54, 2), np.full(54, 0.7), hand_initial_delta=hand_limit)
    machine = StateMachine(run_id="test")
    machine.connect(0)
    machine.ready(0)
    sink = ShadowSink("test")
    scheduler = Scheduler(limits, sink, machine)
    candidate = Candidate("test", 0, 0, "obs0", 0.05, 0.1, 0.2, np.full((50, 54), 0.02), np.zeros(54))
    feedback = Feedback(np.zeros(54), 0.1, 0.24)
    return scheduler, candidate, feedback


def test_messages_copy_inputs_and_map_arms_exactly():
    values = np.arange(54.0)
    frame = OutputFrame("run", 3, 4, 9, values, 1, 2)
    values[:] = -1
    assert frame.positions[27] == 27
    assert not frame.positions.flags.writeable
    payload = arm_payload(frame)
    assert payload["joint_names"] == list(ARM_NAMES)
    assert payload["positions"] == list(np.arange(54.0)[ARM_INDICES])
    assert payload["session_id"] == "run:3"


@pytest.mark.parametrize("actions", [np.zeros((50, 53)), np.full((50, 54), np.nan), np.full((50, 54), np.inf)])
def test_bad_candidate_cannot_enter_queue(actions):
    _, candidate, _ = setup()
    with pytest.raises(ValueError, match="Expected finite"):
        dataclasses.replace(candidate, actions=actions)


def test_single_trial_requires_request_then_stops_and_fences():
    scheduler, candidate, feedback = setup()
    assert scheduler.offer(candidate) is None
    assert scheduler.tick(0.1, feedback) is None
    assert scheduler.sink.submitted == 0
    scheduler.machine.acquire(0.1)
    decision = scheduler.tick(0.11, feedback)
    assert decision["accepted"]
    np.testing.assert_allclose(decision["shadow_target"], 0.007)
    np.testing.assert_allclose(decision["raw_first"], 0.02)
    assert scheduler.machine.state == State.PAUSED
    assert scheduler.machine.generation == 1
    assert scheduler.sink.submitted == 1
    assert scheduler.offer(candidate) == "stale_generation"


def test_jump_rejection_happens_before_slew_and_unknown_hand_limit_blocks():
    scheduler, candidate, feedback = setup(hand_limit=None)
    actions = candidate.actions.copy()
    actions[0, 0] = 0.2
    candidate = dataclasses.replace(candidate, actions=actions)
    scheduler.machine.acquire(0.1)
    scheduler.offer(candidate)
    decision = scheduler.tick(0.1, feedback)
    assert "initial_delta:left_arm" in decision["reasons"]
    assert "hand_acquisition_unconfigured" in decision["reasons"]
    assert scheduler.sink.submitted == 0
    assert scheduler.machine.state == State.READY


def test_expired_feedback_response_and_clock_epoch_never_submit():
    scheduler, candidate, feedback = setup()
    assert "observation_expired" in scheduler.inspect(candidate, feedback, 0.2)["reasons"]
    assert "feedback_expired_or_future" in scheduler.inspect(candidate, feedback, 0.25)["reasons"]
    changed = dataclasses.replace(feedback, clock_epoch=1)
    assert "clock_epoch_changed" in scheduler.inspect(candidate, changed, 0.1)["reasons"]
    scheduler.machine.acquire(0.1)
    scheduler.offer(candidate)
    scheduler.tick(0.25, feedback)
    assert scheduler.machine.state == State.FAULT
    assert scheduler.sink.submitted == 0


def test_newest_candidate_only_duplicate_and_pause_resurrection():
    scheduler, candidate, feedback = setup()
    scheduler.offer(candidate)
    newer = dataclasses.replace(candidate, request_id=2)
    scheduler.offer(newer)
    assert scheduler.pending.request_id == 2
    assert scheduler.offer(candidate) == "duplicate_or_out_of_order_request"
    scheduler.pause(0.1)
    assert scheduler.pending is None
    scheduler.machine.resume(0.12)
    scheduler.machine.ready(0.12)
    assert scheduler.offer(newer) == "stale_generation"
    assert not scheduler.inspect(newer, feedback, 0.12)["accepted"]


def test_queue_exhaustion_and_overrun_do_not_catch_up():
    scheduler, candidate, feedback = setup()
    scheduler.machine.acquire(0.1)
    scheduler.tick(0.1, feedback)
    assert scheduler.machine.state == State.PAUSED
    scheduler, candidate, feedback = setup()
    scheduler.tick(0.1, feedback)
    scheduler.machine.acquire(0.1)
    scheduler.offer(candidate)
    scheduler.tick(0.15, feedback)
    assert scheduler.machine.fault_reason == "scheduler_overrun"
    assert scheduler.sink.submitted == 0
    generation = scheduler.machine.generation
    with pytest.raises(ValueError, match="Cannot"):
        scheduler.machine.resume(0.16)
    assert scheduler.machine.generation == generation


def test_final_sink_checks_deadline_and_duplicate_at_consumption():
    sink = ShadowSink("test")
    frame = OutputFrame("test", 0, 1, 0, np.zeros(54), 0.1, 0.2)
    with pytest.raises(ValueError, match="expired"):
        sink.submit(frame, 0.2)
    sink.submit(frame, 0.15)
    with pytest.raises(ValueError, match="duplicate"):
        sink.submit(frame, 0.16)


def test_disk_delay_does_not_block_control_and_artifact_queue_is_bounded(tmp_path):
    logger = AsyncRecorder(tmp_path, artifact_capacity=1, _write_delay=0.15)
    try:
        t0 = time.monotonic()
        for i in range(20):
            logger.event({"index": i})
            logger.artifact(i, {"state": np.zeros(54)})
        elapsed = time.monotonic() - t0
        assert elapsed < 0.15
        assert logger.artifacts_dropped > 0
        result = logger.close()
        assert result["events"] == 20
        assert result["artifacts"] < 20
        assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 20
    finally:
        logger.close()


def test_hardware_cannot_be_enabled_by_editing_a_flag(tmp_path):
    import yaml

    path = Path(__file__).parent / "executor.example.yaml"
    config = load_config(path)
    for changes in ({"hardware_output": True}, {"mode": "execute"}, {"continuous_execution_enabled": True}):
        target = tmp_path / "bad.yaml"
        target.write_text(yaml.safe_dump({**config, **changes}))
        with pytest.raises(ValueError, match="execution"):
            load_config(target)
