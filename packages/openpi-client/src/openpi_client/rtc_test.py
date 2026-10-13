import copy

import numpy as np
import pytest

from openpi_client import msgpack_numpy
from openpi_client import rtc


def context(*, rows=50, **updates):
    value = {
        "protocol_version": 1,
        "token": "request-0002",
        "source_number": 1,
        "source_sha256": "a" * 64,
        "previous_actions": np.zeros((rows, 54), dtype=np.float32),
        "inference_delay_steps": 4,
        "prefix_attention_horizon": min(25, rows),
        "prefix_attention_schedule": "exp",
        "max_guidance_weight": 10.0,
    }
    value.update(updates)
    return value


def test_raw_observation_is_returned_unchanged():
    observation = {"observation/state": np.zeros(54), "observation": {"nested": True}}
    unwrapped, rtc_context = rtc.unwrap_request(observation)
    assert unwrapped is observation
    assert rtc_context is None


def test_request_envelope_validates_and_survives_msgpack_roundtrip():
    observation = {"observation/state": np.arange(54, dtype=np.float32)}
    request = rtc.make_request(observation, context())
    unpacked = msgpack_numpy.unpackb(msgpack_numpy.packb(request))
    unwrapped, checked = rtc.unwrap_request(unpacked)
    np.testing.assert_array_equal(unwrapped["observation/state"], observation["observation/state"])
    assert checked is not None
    assert checked.token == "request-0002"
    assert checked.previous_actions.shape == (50, 54)
    assert checked.previous_actions.dtype == np.float32


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", 2),
        ("protocol_version", True),
        ("token", ""),
        ("token", " token"),
        ("source_number", 0),
        ("source_number", True),
        ("source_sha256", "A" * 64),
        ("source_sha256", "a" * 63),
        ("inference_delay_steps", -1),
        ("inference_delay_steps", True),
        ("prefix_attention_horizon", 51),
        ("prefix_attention_schedule", "cosine"),
        ("max_guidance_weight", 0),
        ("max_guidance_weight", float("nan")),
        ("max_guidance_weight", 100.01),
    ],
)
def test_invalid_scalar_context_fields_are_rejected(field, value):
    with pytest.raises(rtc.RTCProtocolError):
        rtc.validate_context(context(**{field: value}))


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((50, 53)),
        np.zeros(54),
        np.zeros((0, 54)),
        np.full((50, 54), np.nan),
        np.zeros((50, 54), dtype=np.complex64),
        np.full((50, 54), "not-a-number"),
        [[0] * 54, [0] * 53],
        np.full((50, 54), np.finfo(np.float64).max),
    ],
)
def test_invalid_previous_actions_are_rejected(actions):
    with pytest.raises(rtc.RTCProtocolError):
        rtc.validate_context(context(previous_actions=actions))


def test_horizons_are_bounded_by_available_previous_actions():
    checked = rtc.validate_context(context(rows=6, inference_delay_steps=2, prefix_attention_horizon=6))
    assert checked.previous_actions.shape == (6, 54)
    with pytest.raises(rtc.RTCProtocolError, match="<= 6"):
        rtc.validate_context(context(rows=6, inference_delay_steps=2, prefix_attention_horizon=7))
    with pytest.raises(rtc.RTCProtocolError, match="inference_delay_steps"):
        rtc.validate_context(context(inference_delay_steps=26, prefix_attention_horizon=25))


def test_previous_actions_may_extend_past_fixed_model_horizon():
    checked = rtc.validate_context(context(rows=75, prefix_attention_horizon=50))
    assert checked.previous_actions.shape == (75, 54)


def test_context_and_envelope_require_exact_fields():
    missing = context()
    missing.pop("token")
    with pytest.raises(rtc.RTCProtocolError, match="missing"):
        rtc.validate_context(missing)
    with pytest.raises(rtc.RTCProtocolError, match="extra"):
        rtc.validate_context({**context(), "unexpected": 1})
    with pytest.raises(rtc.RTCProtocolError, match="extra"):
        rtc.validate_context({**context(), 1: "non-string key"})
    with pytest.raises(rtc.RTCProtocolError, match="RTC envelope"):
        rtc.unwrap_request({"observation": {}, "rtc": context(), "unexpected": 1})
    with pytest.raises(rtc.RTCProtocolError, match="RTC envelope"):
        rtc.unwrap_request({"rtc": context()})


def test_reference_prefix_weight_schedules_have_fixed_horizon():
    linear = np.array([1, 1, 0.8, 0.6, 0.4, 0.2] + [0] * 44, dtype=np.float32)
    expected = {
        "ones": np.array([1] * 6 + [0] * 44, dtype=np.float32),
        "zeros": np.array([1] * 2 + [0] * 48, dtype=np.float32),
        "linear": linear,
        "exp": linear * np.expm1(linear) / np.expm1(np.float32(1.0)),
    }
    for schedule, values in expected.items():
        result = rtc.prefix_weights(2, 6, schedule)
        assert result.shape == (50,)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, values, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(rtc.prefix_weights(0, 0, "ones"), np.zeros(50, dtype=np.float32))
    np.testing.assert_allclose(
        rtc.prefix_weights(0, 2, "linear")[:3],
        np.array([2 / 3, 1 / 3, 0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "args",
    [(-1, 4, "exp"), (5, 4, "exp"), (0, 51, "linear"), (0, 4, "wrong"), (True, 4, "ones")],
)
def test_invalid_prefix_weight_arguments_are_rejected(args):
    with pytest.raises(rtc.RTCProtocolError):
        rtc.prefix_weights(*args)


def test_receipt_roundtrip_binds_every_request_field():
    checked_context = rtc.validate_context(context())
    receipt = rtc.make_receipt(checked_context)
    unpacked = msgpack_numpy.unpackb(msgpack_numpy.packb(receipt))
    checked_receipt = rtc.validate_receipt(unpacked, checked_context)
    assert checked_receipt.to_dict() == receipt
    assert "previous_actions" not in receipt


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", 2),
        ("token", "different-request"),
        ("source_number", 2),
        ("source_sha256", "b" * 64),
        ("inference_delay_steps", 3),
        ("prefix_attention_horizon", 24),
        ("prefix_attention_schedule", "linear"),
        ("max_guidance_weight", 9.0),
    ],
)
def test_receipt_mismatch_is_rejected_before_chunk_admission(field, value):
    receipt = rtc.make_receipt(context())
    receipt[field] = value
    with pytest.raises(rtc.RTCProtocolError, match="does not match"):
        rtc.validate_receipt(receipt, context())


def test_receipt_requires_exact_fields_and_valid_internal_constraints():
    receipt = rtc.make_receipt(context())
    missing = copy.copy(receipt)
    missing.pop("token")
    with pytest.raises(rtc.RTCProtocolError, match="missing"):
        rtc.validate_receipt(missing)
    with pytest.raises(rtc.RTCProtocolError, match="extra"):
        rtc.validate_receipt({**receipt, "previous_actions": np.zeros((50, 54))})
    with pytest.raises(rtc.RTCProtocolError, match="must not exceed"):
        rtc.validate_receipt({**receipt, "inference_delay_steps": 26, "prefix_attention_horizon": 25})
