"""Inference admission and cancellation, before any real device acquisition."""

import json
import time
from types import SimpleNamespace

import numpy as np
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
