"""Validated wire helpers for Real-Time Chunking (RTC) inference.

Legacy clients send an observation dictionary directly. RTC clients wrap that
same dictionary in ``{"observation": observation, "rtc": context}``.

The prefix weighting follows Physical Intelligence's reference implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import math
import re
from typing import Any

import numpy as np


PROTOCOL_VERSION = 1
ACTION_HORIZON = 50
ACTION_DIM = 54
RTC_RECEIPT_KEY = "rtc"
PREFIX_ATTENTION_SCHEDULES = frozenset({"ones", "zeros", "linear", "exp"})
MAX_GUIDANCE_WEIGHT = 100.0

_CONTEXT_FIELDS = frozenset(
    {
        "protocol_version",
        "token",
        "source_number",
        "source_sha256",
        "previous_actions",
        "inference_delay_steps",
        "prefix_attention_horizon",
        "prefix_attention_schedule",
        "max_guidance_weight",
    }
)
_RECEIPT_FIELDS = _CONTEXT_FIELDS - {"previous_actions"}
_ENVELOPE_FIELDS = frozenset({"observation", "rtc"})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RTCProtocolError(ValueError):
    """Raised when an RTC request or receipt violates the wire contract."""


@dataclasses.dataclass(frozen=True)
class RTCContext:
    protocol_version: int
    token: str
    source_number: int
    source_sha256: str
    previous_actions: np.ndarray
    inference_delay_steps: int
    prefix_attention_horizon: int
    prefix_attention_schedule: str
    max_guidance_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "token": self.token,
            "source_number": self.source_number,
            "source_sha256": self.source_sha256,
            "previous_actions": self.previous_actions.copy(),
            "inference_delay_steps": self.inference_delay_steps,
            "prefix_attention_horizon": self.prefix_attention_horizon,
            "prefix_attention_schedule": self.prefix_attention_schedule,
            "max_guidance_weight": self.max_guidance_weight,
        }


@dataclasses.dataclass(frozen=True)
class RTCReceipt:
    protocol_version: int
    token: str
    source_number: int
    source_sha256: str
    inference_delay_steps: int
    prefix_attention_horizon: int
    prefix_attention_schedule: str
    max_guidance_weight: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RTCProtocolError(f"{label} must be a mapping")
    return value


def _require_exact_fields(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    fields = set(value)
    missing = sorted(expected - fields, key=repr)
    extra = sorted(fields - expected, key=repr)
    if missing or extra:
        raise RTCProtocolError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _integer(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RTCProtocolError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise RTCProtocolError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise RTCProtocolError(f"{label} must be at most {maximum}")
    return result


def _token(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128 or value.strip() != value:
        raise RTCProtocolError("token must be a non-empty, trimmed string of at most 128 characters")
    return value


def _source_sha256(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RTCProtocolError("source_sha256 must be a lowercase 64-character hexadecimal digest")
    return value


def _schedule(value: Any) -> str:
    if not isinstance(value, str) or value not in PREFIX_ATTENTION_SCHEDULES:
        raise RTCProtocolError(f"prefix_attention_schedule must be one of {sorted(PREFIX_ATTENTION_SCHEDULES)}")
    return value


def _guidance_weight(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RTCProtocolError("max_guidance_weight must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= MAX_GUIDANCE_WEIGHT:
        raise RTCProtocolError(f"max_guidance_weight must be finite and in (0, {MAX_GUIDANCE_WEIGHT:g}]")
    return result


def _previous_actions(value: Any) -> np.ndarray:
    try:
        actions = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise RTCProtocolError("previous_actions must be a rectangular numeric array") from error
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != ACTION_DIM:
        raise RTCProtocolError(f"previous_actions must have shape (K, {ACTION_DIM}) with K >= 1")
    if not np.issubdtype(actions.dtype, np.number) or np.issubdtype(actions.dtype, np.complexfloating):
        raise RTCProtocolError("previous_actions must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        actions = np.array(actions, dtype=np.float32, copy=True)
    if not np.isfinite(actions).all():
        raise RTCProtocolError("previous_actions must contain only finite values")
    return actions


def validate_context(context: RTCContext | Mapping[str, Any]) -> RTCContext:
    """Validate and normalize one RTC request context."""

    if isinstance(context, RTCContext):
        context = context.to_dict()
    value = _require_mapping(context, "rtc context")
    _require_exact_fields(value, _CONTEXT_FIELDS, "rtc context")

    version = _integer(value["protocol_version"], "protocol_version")
    if version != PROTOCOL_VERSION:
        raise RTCProtocolError(f"protocol_version must equal {PROTOCOL_VERSION}")
    actions = _previous_actions(value["previous_actions"])
    delay = _integer(value["inference_delay_steps"], "inference_delay_steps", minimum=0)
    horizon = _integer(value["prefix_attention_horizon"], "prefix_attention_horizon", minimum=0)
    maximum_horizon = min(ACTION_HORIZON, len(actions))
    if not delay <= horizon <= maximum_horizon:
        raise RTCProtocolError(
            "RTC horizons must satisfy "
            f"0 <= inference_delay_steps <= prefix_attention_horizon <= {maximum_horizon}"
        )

    return RTCContext(
        protocol_version=version,
        token=_token(value["token"]),
        source_number=_integer(value["source_number"], "source_number", minimum=1),
        source_sha256=_source_sha256(value["source_sha256"]),
        previous_actions=actions,
        inference_delay_steps=delay,
        prefix_attention_horizon=horizon,
        prefix_attention_schedule=_schedule(value["prefix_attention_schedule"]),
        max_guidance_weight=_guidance_weight(value["max_guidance_weight"]),
    )


def make_request(observation: Mapping[str, Any], context: RTCContext | Mapping[str, Any]) -> dict[str, Any]:
    """Build a validated RTC request envelope."""

    observation = _require_mapping(observation, "observation")
    checked = validate_context(context)
    return {"observation": observation, "rtc": checked.to_dict()}


def unwrap_request(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], RTCContext | None]:
    """Return ``(observation, context)`` for raw and RTC-wrapped requests.

    A mapping without the reserved top-level ``rtc`` field is an unmodified
    legacy observation and is returned by identity.
    """

    payload = _require_mapping(payload, "request")
    if "rtc" not in payload:
        return payload, None
    _require_exact_fields(payload, _ENVELOPE_FIELDS, "RTC envelope")
    observation = _require_mapping(payload["observation"], "observation")
    return observation, validate_context(payload["rtc"])


def prefix_weights(
    inference_delay_steps: int,
    prefix_attention_horizon: int,
    prefix_attention_schedule: str,
) -> np.ndarray:
    """Return the reference RTC prefix weights on the fixed 50-step horizon."""

    delay = _integer(inference_delay_steps, "inference_delay_steps", minimum=0, maximum=ACTION_HORIZON)
    horizon = _integer(
        prefix_attention_horizon,
        "prefix_attention_horizon",
        minimum=0,
        maximum=ACTION_HORIZON,
    )
    if delay > horizon:
        raise RTCProtocolError("inference_delay_steps must not exceed prefix_attention_horizon")
    schedule = _schedule(prefix_attention_schedule)

    indices = np.arange(ACTION_HORIZON, dtype=np.float64)
    if schedule == "ones":
        weights = np.ones(ACTION_HORIZON, dtype=np.float64)
    elif schedule == "zeros":
        weights = (indices < delay).astype(np.float64)
    else:
        weights = np.clip((delay - 1 - indices) / (horizon - delay + 1) + 1, 0, 1)
        if schedule == "exp":
            weights = weights * np.expm1(weights) / np.expm1(1.0)
    weights[indices >= horizon] = 0
    return weights.astype(np.float32)


def make_receipt(context: RTCContext | Mapping[str, Any]) -> dict[str, Any]:
    """Build the response receipt that binds a result to an RTC request."""

    checked = validate_context(context)
    return RTCReceipt(
        protocol_version=checked.protocol_version,
        token=checked.token,
        source_number=checked.source_number,
        source_sha256=checked.source_sha256,
        inference_delay_steps=checked.inference_delay_steps,
        prefix_attention_horizon=checked.prefix_attention_horizon,
        prefix_attention_schedule=checked.prefix_attention_schedule,
        max_guidance_weight=checked.max_guidance_weight,
    ).to_dict()


def validate_receipt(
    receipt: RTCReceipt | Mapping[str, Any],
    expected_context: RTCContext | Mapping[str, Any] | None = None,
) -> RTCReceipt:
    """Validate a response receipt and, when supplied, bind every field to a request."""

    if isinstance(receipt, RTCReceipt):
        receipt = receipt.to_dict()
    value = _require_mapping(receipt, "RTC receipt")
    _require_exact_fields(value, _RECEIPT_FIELDS, "RTC receipt")

    version = _integer(value["protocol_version"], "protocol_version")
    if version != PROTOCOL_VERSION:
        raise RTCProtocolError(f"protocol_version must equal {PROTOCOL_VERSION}")
    delay = _integer(
        value["inference_delay_steps"],
        "inference_delay_steps",
        minimum=0,
        maximum=ACTION_HORIZON,
    )
    horizon = _integer(
        value["prefix_attention_horizon"],
        "prefix_attention_horizon",
        minimum=0,
        maximum=ACTION_HORIZON,
    )
    if delay > horizon:
        raise RTCProtocolError("inference_delay_steps must not exceed prefix_attention_horizon")
    checked = RTCReceipt(
        protocol_version=version,
        token=_token(value["token"]),
        source_number=_integer(value["source_number"], "source_number", minimum=1),
        source_sha256=_source_sha256(value["source_sha256"]),
        inference_delay_steps=delay,
        prefix_attention_horizon=horizon,
        prefix_attention_schedule=_schedule(value["prefix_attention_schedule"]),
        max_guidance_weight=_guidance_weight(value["max_guidance_weight"]),
    )

    if expected_context is not None:
        expected = make_receipt(expected_context)
        actual = checked.to_dict()
        mismatched = [field for field in sorted(_RECEIPT_FIELDS) if actual[field] != expected[field]]
        if mismatched:
            raise RTCProtocolError(f"RTC receipt does not match its request: {mismatched}")
    return checked
