"""Fresh camera inference admission and dry deployment entry-point tests."""

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest

from deployment import slow as deploy
from deployment.profiles import DP3_PROFILE
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
            assert 0 < timeout <= 1.0
            if case == "stale_reply":
                time.sleep(1.01)
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
        with pytest.raises(ValueError, match="1 s|50, 54"):
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
    monkeypatch.setattr(deploy.hardware, "write_runtime", lambda *args, **kwargs: ({"controller_sha256": "test"}, {}, []))

    def forbidden(**kwargs):
        raise AssertionError("Hardware startup was reached without commissioning evidence")

    monkeypatch.setattr(deploy.hardware, "preflight_owners", forbidden)
    with pytest.raises(ValueError, match="requires measured"):
        deploy.main(["--execute", "--output", str(tmp_path / "run")])


def test_launch_omits_legacy_reset_writer_and_keeps_reference_cpu_layout(tmp_path):
    args = deploy.parse_args(["--execute", "--qualification", str(tmp_path / "qualification.json")])
    commands = deploy.hardware.launch_commands(args, tmp_path, tmp_path / "ipc")
    arm = commands["arms"]
    assert arm[arm.index("--cpuset-cpus") + 1] == "8-9,12-13"
    assert "franka_fr3_arm_controllers.launch.py" in " ".join(arm)
    assert "robot_control.launch.py" not in " ".join(arm)
    for role in ("gateway", "splitter", "devices"):
        assert commands[role][commands[role].index("--cpuset-cpus") + 1] == "0-7,16-23"


def test_supervised_trial_explicitly_propagates_without_fake_evidence(tmp_path):
    args = deploy.parse_args(["--execute", "--supervised-trial"])
    commands = deploy.hardware.launch_commands(args, tmp_path, tmp_path / "ipc")
    assert "--supervised-trial" in commands["devices"]
    assert "--qualification" not in commands["devices"]
    assert args.qualification is None
    with pytest.raises(SystemExit):
        deploy.parse_args(["--supervised-trial"])
    with pytest.raises(SystemExit):
        deploy.parse_args(["--execute", "--supervised-trial", "--qualification", "anything.json"])


def test_dp3_four_frame_observation_executes_eight_step_plan(tmp_path, monkeypatch):
    sent, executed = [], []

    class Devices:
        def feedback(self, now):
            return Feedback(np.zeros(54), np.zeros(54), (now,) * 4, (now,) * 4)

    class WS:
        def send(self, packet):
            sent.append(msgpack_numpy.unpackb(packet))

        def recv(self, timeout):
            return msgpack_numpy.packb({"actions": np.full((8, 54), 0.01)})

    def execute(player, devices, **kwargs):
        executed.append(player)
        player.state = "complete"

    monkeypatch.setattr(live, "run_live", execute)
    config = read_config(Path(__file__).parents[1] / "config.yaml", source_hz=20)
    config.update(prediction_steps=8, execution_steps=8)
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0), "speed": np.full(54, 2.0)}
    model = {
        "config": "dp3_bimanual_franka_wuji",
        "observation_steps": 4,
        "action_sample_hz": 20,
        "checkpoint_sha256": "test-only",
    }
    consumer = live.LiveConsumer(
        Devices(), WS(), model, config, limits, [str(i) for i in range(54)], tmp_path,
        SimpleNamespace(event=lambda value: None),
    )
    for index in range(4):
        consumer(
            None,
            {"point_cloud": np.zeros((1024, 3)), "observation/state": np.zeros(54)},
            {"target_stamp": index * 0.05, "samples": {"cam0": {"stamp": time.time() - 0.005}}},
            tmp_path,
        )
    assert len(sent) == len(executed) == consumer.completed == 1
    assert sent[0]["point_cloud"].shape == (4, 1024, 3)
    assert sent[0]["agent_pos"].shape == (4, 54)
    assert executed[0].plan.raw.shape == (8, 54)


def test_dp3_execution_requires_validated_depth_camera_profile():
    with pytest.raises(SystemExit):
        deploy.parse_args(["--execute", "--supervised-trial"], profile=DP3_PROFILE)
    args = deploy.parse_args(
        ["--execute", "--supervised-trial", "--start-cameras"], profile=DP3_PROFILE
    )
    assert args.start_cameras
