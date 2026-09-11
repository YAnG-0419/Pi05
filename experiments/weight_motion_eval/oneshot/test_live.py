"""Fresh camera inference admission and dry deployment entry-point tests."""

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest

from experiments.weight_motion_eval.oneshot import deploy
from experiments.weight_motion_eval.oneshot import live
from experiments.weight_motion_eval.oneshot.core import Feedback
from experiments.weight_motion_eval.planner import read_config


@pytest.mark.parametrize("case", ["success", "stale_input", "feedback_delay", "stale_reply", "invalid_actions"])
def test_live_infers_once_and_rejects_stale_or_invalid_output(tmp_path, monkeypatch, case):
    sent, executed = [], []

    class Devices:
        def feedback(self, now):
            if case == "feedback_delay":
                time.sleep(0.08)
            now = time.monotonic()
            return Feedback(np.zeros(54), np.zeros(54), (now,) * 4, (now,) * 4)

    class WS:
        def send(self, packet):
            sent.append(packet)

        def recv(self, timeout):
            assert 0 < timeout <= 0.2
            if case == "stale_reply":
                time.sleep(0.21)
            shape = (49, 54) if case == "invalid_actions" else (50, 54)
            return msgpack_numpy.packb({"actions": np.full(shape, 0.01)})

    def execute(player, devices, **kwargs):
        executed.append(player)
        player.state = "complete"

    monkeypatch.setattr(live, "run_live", execute)
    config = read_config(Path(__file__).parents[1] / "config.yaml")
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0), "speed": np.full(54, 2.0)}
    consumer = live.LiveConsumer(
        Devices(),
        WS(),
        {"checkpoint_manifest_sha256": "test-only"},
        config,
        limits,
        [str(i) for i in range(54)],
        tmp_path,
        SimpleNamespace(event=lambda value: None),
    )
    obs = {"observation/state": np.zeros(54), "prompt": "test"}
    stamp = time.time() - (0.08 if case == "stale_input" else 0.005)
    metadata = {"samples": {"cam0": {"stamp": stamp}}}
    if case in {"stale_reply", "invalid_actions"}:
        with pytest.raises(ValueError, match="200 ms|50, 54"):
            consumer(None, obs, metadata, tmp_path)
        assert not executed
    else:
        consumer(None, obs, metadata, tmp_path)
        if case == "success":
            consumer(None, obs, metadata, tmp_path)
            assert consumer.completed == 1
            assert len(sent) == 1
            assert len(executed) == 1
        else:
            assert not sent
            assert not executed


def test_missing_qualification_fails_before_any_hardware_start(tmp_path, monkeypatch):
    from experiments.weight_motion_eval import cli

    monkeypatch.setattr(cli, "safe_output", lambda value: Path(value))
    monkeypatch.setattr(deploy, "write_runtime", lambda *args: ({"controller_sha256": "test"}, {}, []))

    def forbidden(**kwargs):
        raise AssertionError("Hardware startup was reached without commissioning evidence")

    monkeypatch.setattr(deploy, "preflight_owners", forbidden)
    with pytest.raises(ValueError, match="requires measured"):
        deploy.main(["--execute", "--output", str(tmp_path / "run")])


def test_launch_omits_legacy_reset_writer_and_keeps_reference_cpu_layout(tmp_path):
    args = deploy.parse_args(["--execute", "--qualification", str(tmp_path / "qualification.json")])
    commands = deploy.launch_commands(args, tmp_path, tmp_path / "ipc")
    arm = commands["arms"]
    assert arm[arm.index("--cpuset-cpus") + 1] == "8-9,12-13"
    assert "franka_fr3_arm_controllers.launch.py" in " ".join(arm)
    assert "robot_control.launch.py" not in " ".join(arm)
    for role in ("gateway", "splitter", "devices"):
        assert commands[role][commands[role].index("--cpuset-cpus") + 1] == "0-7,16-23"


def test_supervised_trial_explicitly_propagates_without_fake_evidence(tmp_path):
    args = deploy.parse_args(["--execute", "--supervised-trial"])
    commands = deploy.launch_commands(args, tmp_path, tmp_path / "ipc")
    assert "--supervised-trial" in commands["devices"]
    assert "--qualification" not in commands["devices"]
    assert args.qualification is None
    with pytest.raises(SystemExit):
        deploy.parse_args(["--supervised-trial"])
    with pytest.raises(SystemExit):
        deploy.parse_args(["--execute", "--supervised-trial", "--qualification", "anything.json"])
