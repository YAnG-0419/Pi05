"""Inference admission and cancellation, before any real device acquisition."""

import json
import time
from types import SimpleNamespace

import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_wuji_fast import deploy
from deploy.fr3_wuji_fast.consumer import FastConsumer
from experiments.weight_motion_eval.oneshot.deploy import ROOT


class WebSocket:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def send(self, packet):
        self.sent.append(packet)

    def recv(self, **kwargs):
        return self.response


def make_consumer(tmp_path, response):
    devices = SimpleNamespace(finish_policy="disable")
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0)}
    return FastConsumer(
        devices,
        WebSocket(response),
        {"checkpoint_manifest_sha256": "weights"},
        limits,
        tmp_path,
        {"rounds": 2},
        ROOT / "experiments/weight_motion_eval/config.yaml",
    )


@pytest.mark.parametrize("kind", ["server_error", "wrong_shape"])
def test_inference_error_signals_stop_before_observation_cleanup(tmp_path, kind, capsys):
    from openpi_client import msgpack_numpy

    response = "model failed" if kind == "server_error" else msgpack_numpy.packb({"actions": np.zeros((50, 64))})
    consumer = make_consumer(tmp_path, response)
    try:
        with pytest.raises((RuntimeError, ValueError), match="server returned|50 x 54"):
            consumer(
                None, {"observation/state": np.zeros(54)}, {"samples": {"camera": {"stamp": time.time()}}}, tmp_path
            )
        assert consumer.stop_flag.is_set()
        assert consumer.process is None
        # Check before close/observe cleanup: the operator already has the cause.
        report = json.loads((tmp_path / "fast-report.json").read_text())
        assert report["state"] == "failed"
        assert report["error"] in capsys.readouterr().err
    finally:
        consumer.close()


def test_stale_observation_is_not_sent_and_request_remains_pending(tmp_path):
    consumer = make_consumer(tmp_path, "unused")
    try:
        consumer(
            None, {"observation/state": np.zeros(54)}, {"samples": {"camera": {"stamp": time.time() - 0.2}}}, tmp_path
        )
        assert not consumer.ws.sent
        assert consumer.request == {"number": 1}
        assert not consumer.stop_flag.is_set()
    finally:
        consumer.close()


def test_unsolicited_rtc_receipt_on_raw_request_is_rejected_without_installing(tmp_path):
    from openpi_client import msgpack_numpy

    context = rtc.validate_context(
        {
            "protocol_version": 1,
            "token": "unexpected-rtc-receipt",
            "source_number": 1,
            "source_sha256": "a" * 64,
            "previous_actions": np.zeros((25, 54), dtype=np.float32),
            "inference_delay_steps": 5,
            "prefix_attention_horizon": 25,
            "prefix_attention_schedule": "exp",
            "max_guidance_weight": 5.0,
        }
    )
    response = msgpack_numpy.packb(
        {"actions": np.zeros((50, 54), dtype=np.float32), rtc.RTC_RECEIPT_KEY: rtc.make_receipt(context)}
    )
    consumer = make_consumer(tmp_path, response)
    try:
        with pytest.raises(ValueError, match="Unexpected RTC receipt"):
            consumer(
                None,
                {"observation/state": np.zeros(54)},
                {"samples": {"camera": {"stamp": time.time()}}},
                tmp_path,
            )
        assert consumer.process is None
        assert consumer.stop_flag.is_set()
    finally:
        consumer.close()


def test_observation_that_expires_during_packet_preparation_is_not_sent(tmp_path, monkeypatch):
    clock = {"monotonic": 100.0, "wall": time.time()}
    fake_time = SimpleNamespace(
        time=lambda: clock["wall"],
        monotonic=lambda: clock["monotonic"],
    )
    monkeypatch.setattr("deploy.fr3_wuji_fast.consumer.time", fake_time)
    consumer = make_consumer(tmp_path, "unused")
    original_codec = consumer.codec

    class SlowCodec:
        def packb(self, payload):
            clock["monotonic"] += 0.08
            clock["wall"] += 0.08
            return original_codec.packb(payload)

    consumer.codec = SlowCodec()
    consumer.options["mode"] = "rtc"
    consumer.request = {
        "event": "infer",
        "number": 2,
        "triggered_at": 100.0,
        "rtc_token": "rtc-request-2",
        "rtc_source_number": 1,
        "rtc_source_sha256": "a" * 64,
        "rtc_reference_time": 100.0,
        "rtc_previous_actions": np.zeros((25, 54)).tolist(),
    }
    try:
        consumer(
            None,
            {"observation/state": np.zeros(54)},
            {"samples": {"camera": {"stamp": clock["wall"]}}},
            tmp_path,
        )
        assert not consumer.ws.sent
        assert consumer.request["number"] == 2
        assert not consumer.stop_flag.is_set()
    finally:
        consumer.close()


def test_rtc_followup_sends_bound_prefix_envelope_and_requires_receipt(tmp_path, monkeypatch):
    from openpi_client import msgpack_numpy

    token = "rtc-request-2"
    source_sha256 = "a" * 64
    reference = np.tile(np.arange(25, dtype=np.float32)[:, None], (1, 54))
    expected_context = rtc.validate_context(
        {
            "protocol_version": 1,
            "token": token,
            "source_number": 1,
            "source_sha256": source_sha256,
            "previous_actions": reference,
            "inference_delay_steps": 5,
            "prefix_attention_horizon": 25,
            "prefix_attention_schedule": "exp",
            "max_guidance_weight": 5.0,
        }
    )
    response = msgpack_numpy.packb(
        {"actions": np.zeros((50, 54), dtype=np.float32), "rtc": rtc.make_receipt(expected_context)}
    )
    consumer = make_consumer(tmp_path, response)
    fake_time = SimpleNamespace(time=time.time, monotonic=lambda: 100.0)
    monkeypatch.setattr("deploy.fr3_wuji_fast.consumer.time", fake_time)
    consumer.options.update(
        mode="rtc",
        rtc_initial_delay_steps=5,
        rtc_delay_buffer_size=10,
        rtc_prefix_attention_schedule="exp",
        rtc_max_guidance_weight=5.0,
    )
    consumer.request = {
        "event": "infer",
        "number": 2,
        "triggered_at": 99.999,
        "rtc_token": token,
        "rtc_source_number": 1,
        "rtc_source_sha256": source_sha256,
        "rtc_reference_time": 99.999,
        "rtc_previous_actions": reference.tolist(),
    }

    class LiveProcess:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            self.alive = False

        def terminate(self):
            self.alive = False

    consumer.process = LiveProcess()
    try:
        consumer(
            None,
            {"observation/state": np.zeros(54)},
            {"samples": {"camera": {"stamp": time.time()}}},
            tmp_path,
        )
        payload = msgpack_numpy.unpackb(consumer.ws.sent[0])
        observation, context = rtc.unwrap_request(payload)
        assert observation["observation/state"].shape == (54,)
        assert context is not None
        assert context.token == token
        assert context.source_sha256 == source_sha256
        assert context.inference_delay_steps == 5
        assert context.previous_actions[0, 0] == pytest.approx(0.03)
        assert list(consumer.rtc_delay_steps) == [5]
        saved = json.loads((tmp_path / "chunk-0002/admission.json").read_text())
        assert saved["rtc"]["receipt"]["token"] == token
    finally:
        consumer.close()


def test_wrong_model_metadata_rejected_before_controllers_start(tmp_path, monkeypatch):
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    output = tmp_path / "wrong-model"
    monkeypatch.setattr(deploy, "safe_output", lambda p: output)
    monkeypatch.setattr(deploy.hardware, "preflight_owners", lambda **k: None)
    # Keep the lock file entirely in the test directory.
    (tmp_path / ".deployment").mkdir()
    monkeypatch.setattr(deploy, "ROOT", tmp_path)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def recv(self, **kwargs):
            return msgpack_numpy.packb({"action_dim": 64, "warmed_up": True})

    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong model must never start a controller")

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **k: Connection())
    monkeypatch.setattr(deploy.hardware.Children, "launch", forbidden)
    with pytest.raises(ValueError, match="Wrong 19999"):
        deploy.main(["--execute", "--supervised-trial"])
    assert not (output / "commands.json").exists()
    assert json.loads((output / "fast-config.json").read_text())["hardware_output"] is True


def test_rtc_requires_declared_protocol_capability_and_guided_warmup():
    expected = {"checkpoint_manifest_sha256": "weights"}
    legacy = {**expected, "warmed_up": True}
    deploy.check_model_metadata(legacy, expected, broker_mode="rtg", rtc_schedule="exp")
    with pytest.raises(ValueError, match="RTC-warmed"):
        deploy.check_model_metadata(legacy, expected, broker_mode="rtc", rtc_schedule="exp")

    rtc_model = {
        **legacy,
        "rtc_supported": True,
        "rtc_protocol_version": 1,
        "rtc_algorithm": "soft_prefix_guidance",
        "rtc_warmed_up": True,
        "rtc_prefix_attention_schedules": ["exp"],
    }
    deploy.check_model_metadata(rtc_model, expected, broker_mode="rtc", rtc_schedule="exp")
