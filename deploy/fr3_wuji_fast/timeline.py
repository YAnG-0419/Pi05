"""Hardware-free action timeline; no implicit retiming of policy playback.

RTG follows Wuji's half-chunk asynchronous inference and latency compensation.
Unlike upstream's cubic broker, smoothing starts at the *actual splice offset*,
anchored to the current command, so latency cannot skip the smoothed prefix.
RTC keeps the same asynchronous timing but delegates continuity to model-side
soft prefix guidance and therefore does not apply the cubic splice again.
"""

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
import math
import uuid

import numpy as np
from openpi_client import rtc

from deploy.fr3_wuji_slow.core import ARM
from deploy.fr3_wuji_slow.core import HAND
from deploy.fr3_wuji_slow.core import Admission
from deploy.fr3_wuji_slow.core import vector
from deploy.fr3_wuji_slow.planner import make_phase
from deploy.fr3_wuji_slow.planner import read_config

from .smoothing import smoothing_settings

SOURCE_HZ = 30
CONTROL_HZ = 100
HORIZON = 50
ARM_SPEED_RAD_S = 1.0


@dataclass
class Chunk:
    number: int
    admission: Admission
    raw: np.ndarray
    actions: np.ndarray
    sha256: str
    projected_hand_values: int
    arm_rate_limited_values: int = 0
    rtc_context: rtc.RTCContext | None = None
    rtc_receipt: dict | None = None


def checked_chunk(raw, admission, number, limits, *, rtc_context=None, rtc_receipt=None):
    admission.check()
    if type(number) is not int or number < 1:
        raise ValueError("Invalid inference number")
    raw = np.array(raw, dtype=np.float64, copy=True)
    if raw.shape != (HORIZON, 54) or not np.isfinite(raw).all():
        raise ValueError("Model must return exactly 50 x 54 finite absolute joint positions")
    lower, upper = vector(limits["lower"]), vector(limits["upper"])
    if np.any(lower >= upper):
        raise ValueError("Invalid device joint limits")
    bad = np.argwhere((raw[:, ARM] < lower[ARM]) | (raw[:, ARM] > upper[ARM]))
    if len(bad):
        step, column = bad[0]
        raise ValueError(f"Arm prediction outside joint limits: step={step}, action_index={ARM[column]}")
    actions = raw.copy()
    # Match the committed slow slider path. Keep raw predictions for diagnosis.
    actions[:, HAND] = np.clip(actions[:, HAND], lower[HAND], upper[HAND])
    projected = int(np.count_nonzero(actions != raw))
    digest = hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest()
    context = None
    if rtc_context is not None:
        context = rtc.validate_context(rtc_context)
        rtc.validate_receipt(rtc_receipt, context)
    elif rtc_receipt is not None:
        raise ValueError("Unexpected RTC receipt without an RTC request")
    return Chunk(number, admission, raw, actions, digest, projected, rtc_context=context, rtc_receipt=rtc_receipt)


def check_arm_speed(actions):
    velocities = np.diff(actions[:, ARM], axis=0) * SOURCE_HZ
    bad = np.argwhere(np.abs(velocities) > ARM_SPEED_RAD_S + 1e-8)
    if len(bad):
        step, column = bad[0]
        raise ValueError(
            f"30 Hz arm speed exceeds device ceiling: step={step}, action_index={ARM[column]}, "
            f"requested={velocities[step, column]:.6f} rad/s, limit={ARM_SPEED_RAD_S:.6f}; "
            "playback was not silently slowed"
        )


def limit_arm_speed(actions):
    """Project arm targets onto the 1 rad/s reachable set on the 30 Hz grid."""
    result = np.array(actions, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 54 or not np.isfinite(result).all():
        raise ValueError("Arm rate limiter requires finite 54-dimensional actions")
    maximum_step = ARM_SPEED_RAD_S / SOURCE_HZ
    changed = 0
    for step in range(1, len(result)):
        delta = result[step, ARM] - result[step - 1, ARM]
        excessive = np.abs(delta) > maximum_step + 1e-12
        changed += int(np.count_nonzero(excessive))
        result[step, ARM] = result[step - 1, ARM] + np.clip(delta, -maximum_step, maximum_step)
    return result, changed


def smooth_prefix(chunk, anchor, count):
    """Cubic Hermite prefix, adapted from Wuji cubic_smooth_prefix (Apache-2.0).

    See UPSTREAM.md for the pinned source and the deliberate splice difference.
    """
    result = np.array(chunk, dtype=np.float64, copy=True)
    n = min(count, len(result))
    if n < 2:
        raise ValueError("Splice requires at least two remaining actions")
    end = result[n - 1].copy()
    velocity = result[n] - end if n < len(result) else end - result[n - 2]
    u = np.linspace(0.0, 1.0, n)[:, None]
    result[:n] = (2 * u**3 - 3 * u**2 + 1) * anchor + (-2 * u**3 + 3 * u**2) * end + (u**3 - u**2) * velocity * (n - 1)
    result[0], result[n - 1] = anchor, end
    return result


def sample_nodes(actions, elapsed):
    """Linear interpolation on the original grid; hold the last node one tick."""
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid policy time")
    position = elapsed * SOURCE_HZ
    index = min(int(position + 1e-10), len(actions) - 1)
    if index == len(actions) - 1:
        return actions[-1].copy(), np.zeros(54)
    fraction = min(1.0, max(0.0, position - index))
    delta = actions[index + 1] - actions[index]
    return actions[index] + fraction * delta, delta * SOURCE_HZ


def future_nodes(actions, elapsed=0.0):
    """Sample the finite remainder of a 30 Hz plan from an arbitrary time."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 54 or not np.isfinite(actions).all():
        raise ValueError("RTC reference plan must contain finite 54-dimensional actions")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid RTC reference time")
    count = max(0, math.ceil(len(actions) - elapsed * SOURCE_HZ - 1e-10))
    if count == 0:
        return np.empty((0, 54), dtype=actions.dtype)
    return np.stack([sample_nodes(actions, elapsed + step / SOURCE_HZ)[0] for step in range(count)])


def _plan_sha256(actions):
    return hashlib.sha256(np.asarray(actions, dtype="<f8").tobytes()).hexdigest()


@dataclass
class Initial:
    chunk: Chunk
    start: np.ndarray
    approach: object
    digest: str
    prepared_at: float


def prepare_initial(chunk, start, limits, config_path, now):
    start = vector(start)
    if np.any(start < limits["lower"]) or np.any(start > limits["upper"]):
        raise ValueError("Measured initial state outside joint limits")
    limited, changed = limit_arm_speed(chunk.actions)
    chunk = replace(chunk, actions=limited, arm_rate_limited_values=changed)
    check_arm_speed(chunk.actions)
    config = read_config(config_path)
    config = dict(config, hand_control="slider")
    if np.max(np.abs(start[ARM] - chunk.actions[0, ARM])) > config["arm_raw_initial_delta_rad"]:
        raise ValueError("Initial arm displacement exceeds existing acquisition bound")
    approach = make_phase("approach", np.array([0.0, 1.0]), np.stack([start, chunk.actions[0]]), config, limits)
    # Slider needs time to traverse its first target at its inherited 1 rad/s.
    approach.scale = max(approach.scale, float(np.max(np.abs(start[HAND] - chunk.actions[0, HAND]))))
    if approach.duration > 60:
        raise ValueError("Initial approach exceeds 60 seconds")
    binding = json.dumps(
        {
            "initial": chunk.sha256,
            "request": chunk.admission.request_id,
            "source_hz": SOURCE_HZ,
            "control_hz": CONTROL_HZ,
            "arm_speed_rad_s": ARM_SPEED_RAD_S,
            "arm_smoothing": smoothing_settings(),
            "config": config,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(binding.encode() + start.astype("<f8").tobytes()).hexdigest()
    return Initial(chunk, start, approach, digest, now)


class Timeline:
    """One bounded stream with monotonic chunk numbers and one pending request."""

    def __init__(
        self,
        first,
        *,
        rounds,
        mode="rtg",
        guidance_steps=3,
        trigger_fraction=0.5,
        rtc_initial_delay_steps=5,
        rtc_delay_buffer_size=10,
        rtc_prefix_attention_schedule="exp",
        rtc_max_guidance_weight=5.0,
    ):
        if type(rounds) is not int or not 1 <= rounds <= 1000 or mode not in {"rtg", "rtc", "serial"}:
            raise ValueError("Invalid stream mode/round count")
        if type(guidance_steps) is not int or not 2 <= guidance_steps <= 10:
            raise ValueError("Guidance steps must be 2..10")
        if not math.isfinite(trigger_fraction) or not 0.1 <= trigger_fraction <= 0.9:
            raise ValueError("Trigger fraction must be 0.1..0.9")
        trigger_step = max(1, int(HORIZON * trigger_fraction))
        if (
            type(rtc_initial_delay_steps) is not int
            or not 0 <= rtc_initial_delay_steps <= HORIZON - trigger_step
        ):
            raise ValueError("RTC initial delay must fit inside the prefix attention horizon")
        if type(rtc_delay_buffer_size) is not int or not 1 <= rtc_delay_buffer_size <= 100:
            raise ValueError("RTC delay buffer size must be 1..100")
        if rtc_prefix_attention_schedule not in {"ones", "zeros", "linear", "exp"}:
            raise ValueError("Unsupported RTC prefix attention schedule")
        if (
            isinstance(rtc_max_guidance_weight, bool)
            or not math.isfinite(rtc_max_guidance_weight)
            or not 0 < rtc_max_guidance_weight <= 100
        ):
            raise ValueError("RTC maximum guidance weight must be finite and in (0, 100]")
        self.first, self.rounds, self.mode = first, rounds, mode
        self.guidance_steps = guidance_steps
        self.trigger_step = trigger_step
        self.rtc_initial_delay_steps = rtc_initial_delay_steps
        self.rtc_delay_buffer_size = rtc_delay_buffer_size
        self.rtc_prefix_attention_schedule = rtc_prefix_attention_schedule
        self.rtc_max_guidance_weight = float(rtc_max_guidance_weight)
        self.actions = first.actions.copy()
        self.number, self.offset = 1, 0
        self.started = None
        self.pending_since = None
        self.pending_rtc = None
        self.last_request_id = first.admission.request_id
        self.current_source_sha256 = _plan_sha256(self.actions)

    def start(self, now):
        if self.started is not None:
            raise ValueError("A stream cannot restart")
        self.started = now

    def sample(self, now):
        if self.started is None or now < self.started:
            raise ValueError("Stream has not started or clock reversed")
        return sample_nodes(self.actions, now - self.started)

    def exhausted(self, now):
        return now - self.started >= len(self.actions) / SOURCE_HZ - 1e-10

    def request_due(self, now):
        if self.pending_since is not None or self.number >= self.rounds:
            return False
        step = self.offset + int((now - self.started) * SOURCE_HZ + 1e-10)
        return step >= self.trigger_step if self.mode in {"rtg", "rtc"} else self.exhausted(now)

    def request(self, now):
        if not self.request_due(now):
            raise ValueError("Unexpected inference trigger")
        event = {"event": "infer", "number": self.number + 1, "triggered_at": now}
        pending_rtc = None
        if self.mode == "rtc":
            previous = future_nodes(self.actions, now - self.started)
            maximum_horizon = HORIZON - self.trigger_step
            previous = previous[:maximum_horizon]
            if len(previous) < self.rtc_initial_delay_steps:
                raise ValueError("RTC reference plan is shorter than the configured initial delay")
            pending_rtc = {
                "token": uuid.uuid4().hex,
                "source_number": self.number,
                "source_sha256": self.current_source_sha256,
            }
            event.update(
                rtc_token=pending_rtc["token"],
                rtc_source_number=pending_rtc["source_number"],
                rtc_source_sha256=pending_rtc["source_sha256"],
                rtc_reference_time=now,
                rtc_previous_actions=previous.tolist(),
            )
        self.pending_since = now
        self.pending_rtc = pending_rtc
        return event

    def install(self, chunk, now, limits):
        if self.pending_since is None or chunk.number != self.number + 1:
            raise ValueError("Unexpected, duplicate or out-of-order prediction")
        a = chunk.admission
        a.check()
        if (
            a.request_id == self.last_request_id
            or a.checkpoint != self.first.admission.checkpoint
            or a.epoch != self.first.admission.epoch
            or a.sent_time < self.pending_since
            or not a.received_time <= now <= a.observation_time + 1.0
        ):
            raise ValueError("Foreign, stale or superseded prediction at splice")
        offset = (
            math.ceil(max(0.0, now - a.sent_time) * SOURCE_HZ - 1e-10)
            if self.mode in {"rtg", "rtc"}
            else 0
        )
        if offset > HORIZON - self.guidance_steps:
            raise ValueError("Inference left too few future actions")
        context = None
        if self.mode == "rtc":
            if self.pending_rtc is None or chunk.rtc_context is None:
                raise ValueError("RTC prediction is missing its request binding")
            context = chunk.rtc_context
            rtc.validate_receipt(chunk.rtc_receipt, context)
            if (
                context.token != self.pending_rtc["token"]
                or context.source_number != self.pending_rtc["source_number"]
                or context.source_sha256 != self.pending_rtc["source_sha256"]
                or context.source_number != self.number
                or context.source_sha256 != self.current_source_sha256
            ):
                raise ValueError("Foreign, stale or superseded RTC prefix")
            if offset > context.prefix_attention_horizon:
                raise ValueError("Actual RTC delay exceeded the conditioned prefix horizon")
            # Model-side guidance owns continuity in RTC mode. Keep actual
            # latency skipping and downstream safety projections, but do not
            # apply the RTG cubic splice a second time.
            actions = chunk.actions[offset:].copy()
        else:
            anchor, _ = self.sample(now)
            actions = smooth_prefix(chunk.actions[offset:], anchor, self.guidance_steps)
        if np.any(actions < limits["lower"]) or np.any(actions > limits["upper"]):
            raise ValueError(f"{self.mode.upper()} splice exceeds joint limits")
        actions, rate_limited = limit_arm_speed(actions)
        check_arm_speed(actions)
        # All checks precede replacement; failed predictions cannot leak a frame.
        self.actions, self.offset, self.number = actions, offset, chunk.number
        self.started, self.pending_since, self.pending_rtc = now, None, None
        self.last_request_id = a.request_id
        self.current_source_sha256 = _plan_sha256(actions)
        event = {
            "event": "chunk_started",
            "number": chunk.number,
            "at": now,
            "latency_offset_steps": offset,
            "sha256": chunk.sha256,
            "projected_hand_values": chunk.projected_hand_values,
            "arm_rate_limited_values": chunk.arm_rate_limited_values + rate_limited,
            "executed_nodes": actions.tolist(),
        }
        if context is not None:
            event["rtc"] = {
                "token": context.token,
                "source_number": context.source_number,
                "source_sha256": context.source_sha256,
                "predicted_delay_steps": context.inference_delay_steps,
                "actual_delay_steps": offset,
                "prefix_attention_horizon": context.prefix_attention_horizon,
                "prefix_attention_schedule": context.prefix_attention_schedule,
                "max_guidance_weight": context.max_guidance_weight,
            }
        return event

    def check_pending(self, now):
        if self.pending_since is not None and now - self.pending_since > 1.5:
            raise RuntimeError("Fresh prediction unavailable within 1.5 seconds of trigger")
