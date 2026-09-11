"""Local model transport, timing and failure integration; no hardware connections."""

from contextlib import contextmanager
import json
import threading
import time

from executor.async_recorder import AsyncRecorder
from executor.async_recorder import RecordingError
from executor.inference_worker import InferenceWorker
from executor.inference_worker import request_timing
from executor.live import LiveShadow
from executor.protocol import Feedback
from executor.scheduler import Limits
from executor.state_machine import State
import numpy as np
from openpi_client import msgpack_numpy
import pytest
from websockets.sync.server import serve


@contextmanager
def mock_server():
    def handler(ws):
        packer = msgpack_numpy.Packer()
        ws.send(
            packer.pack(
                {
                    "config": "pi05_fr3_wuji",
                    "action_dim": 54,
                    "action_horizon": 50,
                    "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
                    "hardware_output": False,
                }
            )
        )
        for payload in ws:
            observation = msgpack_numpy.unpackb(payload)
            assert observation["observation/state"].shape == (54,)
            time.sleep(0.1)
            ws.send(packer.pack({"actions": np.full((50, 54), 0.02)}))

    server = serve(handler, "127.0.0.1", 0, compression=None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
    finally:
        server.shutdown()
        thread.join(3)


def test_ipc_and_serialization_cannot_refresh_old_input():
    request = {"capture_wall": 100.0, "capture_mono": 1.0, "metadata": {"samples": {"camera": {"stamp": 99.95}}}}
    assert request_timing(request, 100.01, 1.01)[0] is None
    assert request_timing(request, 100.03, 1.03)[0] == "input_exceeds_send_budget"
    assert request_timing(request, 99.9, 1.04)[0] == "clock_jump"


@pytest.mark.parametrize("pause", [False, True])
def test_model_worker_and_live_coordinator_keep_controls_responsive(tmp_path, monkeypatch, pause):
    limits = Limits(np.full(54, -2), np.full(54, 2), np.full(54, 0.7), hand_initial_delta=0.05)
    monkeypatch.setattr("executor.live.make_limits", lambda _: limits)
    with mock_server() as uri:
        worker = InferenceWorker(uri)
        consumer = LiveShadow({}, worker, single_step=not pause)

        def feedback(_):
            now = time.monotonic()
            return Feedback(np.zeros(54), now, now + 0.15)

        monkeypatch.setattr(consumer, "_feedback", feedback)
        try:
            consumer.poll(None, tmp_path)
            now = time.time()
            metadata = {"target_stamp": now, "samples": {"camera": {"stamp": now}}}
            consumer(None, {"observation/state": np.zeros(54)}, metadata, tmp_path)
            assert worker.pending
            if pause:
                consumer.control("pause")
                consumer.poll(None, tmp_path)
                assert consumer.machine.state == State.PAUSED
                assert consumer.completed == 0
                assert worker.pending
                consumer.control("prepare")
                consumer.poll(None, tmp_path)
                # Resume prepares a new generation. A delayed old response must
                # still be rejected even with healthy inputs again.
                consumer.machine.ready(time.monotonic())
            deadline = time.monotonic() + 3
            while consumer.completed == 0 and time.monotonic() < deadline:
                consumer.poll(None, tmp_path)
                time.sleep(0.005)
            assert consumer.completed == 1
            if pause:
                assert consumer.reasons["stale_generation"] == 1
                assert consumer.scheduler.sink.submitted == 0
            else:
                assert consumer.scheduler.sink.submitted == 1
                assert consumer.machine.state == State.PAUSED
                np.testing.assert_allclose(consumer.scheduler.sink.frames[-1].positions, 0.007)
        finally:
            worker.close()
            consumer.finish(tmp_path)
    report = json.loads((tmp_path / "client-report.json").read_text())
    assert report["status"] == "completed"
    assert report["hardware_output"] is False
    assert report["logging"]["artifacts"] == 1


def test_dead_worker_is_detected_and_logger_failure_faults_client(tmp_path, monkeypatch):
    limits = Limits(np.full(54, -2), np.full(54, 2), np.full(54, 0.7), hand_initial_delta=0.05)
    monkeypatch.setattr("executor.live.make_limits", lambda _: limits)
    with mock_server() as uri:
        worker = InferenceWorker(uri)
        try:
            worker.process.terminate()
            worker.process.join(3)
            with pytest.raises(RuntimeError, match="exited"):
                worker.poll()
            consumer = LiveShadow({}, worker)
            consumer.logger = AsyncRecorder(tmp_path)
            consumer.logger.process.terminate()
            consumer.logger.process.join(3)
            with pytest.raises(RecordingError):
                consumer.poll(None, tmp_path)
            assert consumer.machine.state == State.FAULT
            assert consumer.scheduler.sink.submitted == 0
            with pytest.raises(RecordingError):
                consumer.finish(tmp_path, "injected logger death")
        finally:
            worker.close()
    assert json.loads((tmp_path / "client-report.json").read_text())["status"] == "failed"
