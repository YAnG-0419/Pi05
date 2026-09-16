"""Diagnostic and stale-input regression tests. No hardware connections."""

import time

import numpy as np
from observation import ObservationBuffer
from observation import ObservationUnavailableError
from openpi_client import msgpack_numpy
import pytest
from record_inference import GROUPS
from record_inference import Recorder
from record_inference import analyze
from record_inference import latest_state
from record_inference import load_limits
from record_inference import validate_actions
from test_observation import populate


def test_limits_use_named_hand_joints_and_existing_arm_gateway():
    limits = load_limits()
    assert limits["lower"].shape == (54,)
    assert limits["lower"][7] == -1.187
    assert limits["upper"][8] == 0.698
    np.testing.assert_equal(limits["lower"][:7], limits["lower"][27:34])
    assert limits["arm_initial_delta"] == 0.05
    assert limits["speed"][0] == 0.7


def test_range_jump_and_nominal_speed_are_separate_diagnostics():
    limits = load_limits()
    state = (limits["lower"] + limits["upper"]) / 2
    actions = np.tile(state, (50, 1))
    actions[0, 0] += 0.06  # In range, but acquisition and 30 Hz slew fail.
    actions[20, 7] = limits["upper"][7] + 0.1
    result = analyze(actions, state, state, state, limits)
    assert result["left_arm"]["range_violation_values"] == 0
    assert result["left_arm"]["initial_delta_exceeds_gateway_at_send"]
    assert result["left_arm"]["nominal_speed_exceedance_values"] == 1
    assert result["left_hand"]["range_violation_joints"] == ["l_thumb_cmc_flex"]
    assert result["left_hand"]["range_violation_values"] == 1
    assert result["right_arm"]["first_delta_return_state_max_rad"] == 0
    changed = state.copy()
    changed[GROUPS["right_arm"]] += 0.1
    assert analyze(actions, state, state, changed, limits)["right_arm"]["initial_delta_exceeds_gateway_at_return"]
    assert analyze(actions, state, state, None, limits)["right_arm"]["first_delta_return_state_max_rad"] is None


@pytest.mark.parametrize("actions", [np.zeros((49, 54)), np.full((50, 54), np.nan), np.full((50, 54), np.inf)])
def test_bad_actions_rejected(actions):
    with pytest.raises(ValueError, match="finite"):
        validate_actions(actions)


def test_latest_state_does_not_need_fresh_camera_but_requires_fresh_feedback():
    store = ObservationBuffer()
    populate(store)
    store.buffers["cam0"].clear()
    state, _ = latest_state(store, now_wall=100, now_mono=10)
    assert state.shape == (54,)
    with pytest.raises(ObservationUnavailableError, match="stale"):
        latest_state(store, now_wall=100.2, now_mono=10.2)
    with pytest.raises(ObservationUnavailableError, match="clock_jump"):
        latest_state(store, now_wall=101, now_mono=10)


def test_expired_input_never_reaches_websocket(tmp_path):
    class NoSend:
        def send(self, _):
            pytest.fail("Stale input must never be sent")

    store = ObservationBuffer()
    populate(store, wall=time.time() - 1, mono=time.monotonic() - 1)
    obs, metadata = store.snapshot(now_wall=time.time() - 1, now_mono=time.monotonic() - 1)
    recorder = Recorder(NoSend(), load_limits())
    recorder(store, obs, metadata, tmp_path)
    assert recorder.completed == 0
    assert recorder.rejected["input_expired_before_send"] == 1


def test_return_expiry_is_independent_of_new_feedback(tmp_path, monkeypatch):
    import record_inference

    clock = {"wall": 100.0, "mono": 10.0}
    monkeypatch.setattr(record_inference.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(record_inference.time, "monotonic", lambda: clock["mono"])
    store = ObservationBuffer()
    populate(store)
    obs, metadata = store.snapshot()

    class DelayedReply:
        def send(self, request):
            assert msgpack_numpy.unpackb(request)["observation/state"].shape == (54,)

        def recv(self, timeout):
            clock.update(wall=100.12, mono=10.12)
            # New feedback is available, but must not refresh the original input.
            populate(store, wall=100.12, mono=10.12)
            return msgpack_numpy.packb({"actions": np.tile(obs["observation/state"], (50, 1))})

    recorder = Recorder(DelayedReply(), load_limits())
    recorder(store, obs, metadata, tmp_path)
    assert recorder.completed == 1
    run = recorder.runs[0]
    assert run["input_expired_at_return"]
    assert run["return_state_error"] is None
    assert max(run["input_age_at_return_seconds"].values()) == pytest.approx(0.22)
    assert max(v["source_age_seconds"] for v in run["latest_state_at_return"].values()) == pytest.approx(0.1)


def test_inference_budget_waits_for_newer_input_without_relaxing_limits(tmp_path, monkeypatch):
    import record_inference

    monkeypatch.setattr(record_inference.time, "time", lambda: 100.0)
    monkeypatch.setattr(record_inference.time, "monotonic", lambda: 10.0)
    store = ObservationBuffer()
    populate(store)
    obs, metadata = store.snapshot()

    class NoSend:
        def send(self, _):
            pytest.fail("100 ms input leaves too little inference time for this 70 ms budget")

    recorder = Recorder(NoSend(), load_limits(), max_input_age=0.07)
    recorder(store, obs, metadata, tmp_path)
    assert recorder.completed == 0
    assert recorder.rejected["input_exceeds_inference_age_budget"] == 1
    assert store.max_age == 0.2
    assert store.state_live_age == 0.15
    assert store.max_skew == 0.08
