"""Offline trajectory mathematics and failure behaviour; no device imports."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.weight_motion_eval.cli import safe_output
from experiments.weight_motion_eval.planner import build_plan
from experiments.weight_motion_eval.planner import caps
from experiments.weight_motion_eval.planner import extrema
from experiments.weight_motion_eval.planner import load_record
from experiments.weight_motion_eval.planner import make_phase
from experiments.weight_motion_eval.planner import read_config
from experiments.weight_motion_eval.planner import rest_spline
from experiments.weight_motion_eval.planner import save_plan
from experiments.weight_motion_eval.preview import write_preview
from experiments.weight_motion_eval.rehearsal import Rehearsal
from experiments.weight_motion_eval.rehearsal import State
from experiments.weight_motion_eval.rehearsal import run_rehearsal


@pytest.fixture
def inputs():
    config = read_config(Path(__file__).with_name("config.yaml"))
    limits = {"lower": np.full(54, -2.0), "upper": np.full(54, 2.0), "speed": np.full(54, 2.0)}
    raw = np.repeat((0.08 + 0.02 * np.sin(np.linspace(0, 2 * np.pi, 50)))[:, None], 54, axis=1)
    return raw, np.zeros(54), config, limits, [f"joint{i}" for i in range(54)]


def test_quintic_extrema_match_known_rest_to_rest_solution():
    curve = rest_spline([0.0, 1.0], np.stack([np.zeros(54), np.ones(54)]))
    low, high = extrema(curve)
    np.testing.assert_allclose(high[1], 1.875, atol=1e-10)
    np.testing.assert_allclose(high[2], 10 * np.sqrt(3) / 3, atol=1e-10)
    np.testing.assert_allclose(low[2], -10 * np.sqrt(3) / 3, atol=1e-10)
    np.testing.assert_allclose(high[3], 60, atol=1e-10)


def test_continuous_bounds_endpoints_and_shared_time_scale(inputs):
    raw = inputs[0].copy()
    plan = build_plan(*inputs)
    np.testing.assert_array_equal(plan.raw, raw)
    np.testing.assert_array_equal(plan.knots[[0, -1]], raw[[0, -1]])
    np.testing.assert_allclose(plan.approach.sample(0), plan.start, atol=1e-12)
    np.testing.assert_allclose(plan.approach.sample(plan.approach.duration), raw[0], atol=1e-12)
    np.testing.assert_allclose(plan.playback.sample(0), raw[0], atol=1e-12)
    np.testing.assert_allclose(plan.playback.sample(plan.playback.duration), raw[-1], atol=1e-12)
    for phase in (plan.approach, plan.playback):
        for order, name in enumerate(("velocity", "acceleration", "jerk"), 1):
            peak = np.maximum(np.abs(phase.lower[order]), np.abs(phase.upper[order])) / phase.scale**order
            assert np.all(peak <= caps(plan.config, phase.name, name) + 1e-10)
        for order in (1, 2):
            np.testing.assert_allclose(phase.sample([0, phase.duration], order), 0, atol=1e-10)
    assert plan.report["hardware_ready"] is False
    with pytest.raises(ValueError, match="lifetime"):
        plan.playback.sample(plan.playback.duration + 0.01)


def test_more_time_reduces_derivatives_without_changing_path(inputs):
    plan = build_plan(*inputs)
    changed = copy.deepcopy(inputs)
    changed[2]["minimum_time_scale"] = plan.playback.scale * 2
    slower = build_plan(*changed)
    for order in range(4):
        expected = plan.playback.sample(plan.playback.duration * 0.37, order) / 2**order
        np.testing.assert_allclose(slower.playback.sample(slower.playback.duration * 0.37, order), expected, atol=1e-7)


def test_default_plan_slows_50_degree_hand_ramp_to_30_without_truncating(inputs):
    raw, start, config, limits, names = inputs
    raw = np.zeros_like(raw)
    raw[:, 7:27] = np.deg2rad(50) * np.arange(50)[:, None] / 30
    raw[:, 34:54] = -raw[:, 7:27]
    plan = build_plan(raw, start, config, limits, names)
    assert config["approach"]["hand"]["velocity"] == pytest.approx(np.deg2rad(15))
    assert config["playback"]["hand"]["velocity"] == pytest.approx(np.deg2rad(30))
    assert plan.playback.duration > 49 / 30
    for group in ("left_hand", "right_hand"):
        assert plan.report["groups"][group]["raw_velocity_max_rad_s"] == pytest.approx(np.deg2rad(50))
        assert plan.report["groups"][group]["playback"]["velocity_max"] <= np.deg2rad(30)
    np.testing.assert_allclose(plan.playback.sample(plan.playback.duration), raw[-1], atol=1e-10)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 3.0])
def test_invalid_or_out_of_range_raw_predictions_are_rejected(inputs, bad):
    inputs[0][20, 0] = bad
    with pytest.raises(ValueError, match="finite|limits"):
        build_plan(*inputs)


def test_large_raw_acquisition_is_rejected_before_smoothing(inputs):
    inputs[0][0, 0] = 0.201
    with pytest.raises(ValueError, match="raw initial delta"):
        build_plan(*inputs)


def test_spline_overshoot_is_rejected_even_when_knots_are_in_bounds(inputs):
    limits = inputs[3]
    limits["lower"][:] = 0
    limits["upper"][:] = 1
    values = np.repeat(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])[:, None], 54, axis=1)
    with pytest.raises(ValueError, match="continuous spline"):
        make_phase("playback", np.arange(7), values, inputs[2], limits)


@pytest.mark.parametrize(
    ("inject", "state"),
    [
        (None, "complete"),
        ("pause", "paused"),
        ("feedback_loss", "fault"),
        ("scheduler_delay", "fault"),
        ("tracking_error", "fault"),
    ],
)
def test_rehearsal_waits_for_settling_and_stops_on_faults(inputs, inject, state):
    plan = build_plan(*inputs)
    report, arrays = run_rehearsal(plan, inject=inject)
    assert report["state"] == state
    assert report["hardware_output"] is False
    assert len(arrays["time"]) == report["frames"]
    if inject is None:
        assert [t["to"] for t in report["transitions"]] == [
            "approach",
            "settle_start",
            "playback",
            "settle_end",
            "complete",
        ]


def test_pause_cannot_restart_old_plan_and_start_position_must_match(inputs):
    plan = build_plan(*inputs)
    engine = Rehearsal(plan)
    engine.start(0, plan.start + 0.02)
    assert engine.state == State.FAULT
    assert engine.tick(0.01, plan.start, np.zeros(54), 0.01) is None
    engine = Rehearsal(plan)
    engine.start(0, plan.start)
    engine.pause(0.1)
    with pytest.raises(ValueError, match="only be started once"):
        engine.start(0.2, plan.start)


def test_hardware_flag_and_unsafe_output_paths_are_rejected(inputs):
    inputs[2]["hardware_output"] = True
    with pytest.raises(ValueError, match="Hardware output"):
        build_plan(*inputs)
    with pytest.raises(ValueError, match="Output must"):
        safe_output("deploy/fr3_wuji/changed")


def test_artifacts_preserve_raw_and_unknown_hand_constraints(inputs, tmp_path):
    plan = build_plan(*inputs)
    output = tmp_path / "plan"
    save_plan(plan, output)
    write_preview(plan, output / "preview.html")
    report = json.loads((output / "report.json").read_text())
    assert "hand_raw_acquisition_threshold" in report["unverified"]
    assert len(report["artifact_sha256"]) == 64
    with np.load(output / "plan.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["raw_actions"], inputs[0])
        assert arrays["playback/velocity"].shape[1] == 54
    assert '<script id="data"' in (output / "preview.html").read_text()
    with pytest.raises(FileExistsError):
        save_plan(plan, output)


def test_record_requires_actual_recorded_feedback(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, actions=np.zeros((50, 54)), **{"observation/state": np.zeros(54)})
    with pytest.raises(ValueError, match="state_at_return"):
        load_record(path)


@pytest.mark.parametrize("expired", [False, True])
def test_capture_requests_exactly_once_and_preserves_stale_failure(tmp_path, monkeypatch, expired):
    from openpi_client import msgpack_numpy

    from experiments.weight_motion_eval.capture import capture_once

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def recv(self, **_):
            return msgpack_numpy.Packer().pack(
                {
                    "config": "pi05_fr3_wuji",
                    "action_dim": 54,
                    "action_horizon": 50,
                    "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
                    "hardware_output": False,
                }
            )

    class Recorder:
        def __init__(self, *_args, **_kwargs):
            self.completed = 0
            self.runs = []

        def __call__(self, *_):
            self.completed += 1
            self.runs.append({"input_expired_at_return": expired, "return_state_error": None})

    def observe(args, *, consumer):
        assert args[args.index("--arm-source") + 1] == "ros"
        output.mkdir()
        for _ in range(5):
            consumer(None, None, None, output)
        assert consumer.completed == 1

    output = tmp_path / "capture"
    monkeypatch.setattr("websockets.sync.client.connect", lambda *_args, **_kwargs: Socket())
    module = SimpleNamespace(Recorder=Recorder, load_limits=dict, observe=SimpleNamespace(main=observe))
    monkeypatch.setattr("experiments.weight_motion_eval.capture.deployment_module", lambda _: module)
    if expired:
        with pytest.raises(RuntimeError, match="stale"):
            capture_once(output, uri="ws://unused")
    else:
        capture_once(output, uri="ws://unused")
    report = json.loads((output / "capture-report.json").read_text())
    assert report["status"] == ("failed" if expired else "captured")
    assert report["hardware_output"] is False
