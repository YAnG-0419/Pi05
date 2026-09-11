"""Two rest-to-rest trajectories with analytically checked spline extrema."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PPoly
from scipy.interpolate import make_interp_spline
import yaml

GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 27), "right_arm": slice(27, 34), "right_hand": slice(34, 54)}


def checked_array(value, shape):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {shape} array")
    return value


def positive(value):
    return isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value) and value > 0


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration")
    if config.get("hardware_output") is not False:
        raise ValueError("Hardware output is unavailable in this tool")
    if config.get("source_hz") != 30:
        raise ValueError("The recorded policy uses 30 Hz")
    for key in ("sample_hz", "minimum_time_scale", "max_total_seconds", "settle_seconds", "arm_raw_initial_delta_rad"):
        if not positive(config.get(key)):
            raise ValueError(f"Invalid positive configuration: {key}")
    if not 10 <= config["sample_hz"] <= 200 or not 1 <= config["minimum_time_scale"] <= 100:
        raise ValueError("Sampling or time scale out of supported range")
    if config["max_total_seconds"] > 120 or config["arm_raw_initial_delta_rad"] > 0.2:
        raise ValueError("This experiment supports at most 120 seconds and 0.2 rad raw arm acquisition")
    passes = config.get("smoothing_passes")
    if type(passes) is not int or not 0 <= passes <= 10:
        raise ValueError("Smoothing passes must be an integer between 0 and 10")
    hand_delta = config.get("hand_raw_initial_delta_rad")
    if hand_delta is not None and not positive(hand_delta):
        raise ValueError("Invalid hand raw acquisition threshold")
    for part in ("arm", "hand"):
        if not positive(config.get("max_smoothing_delta_rad", {}).get(part)):
            raise ValueError("Invalid smoothing deviation bound")
        for phase in ("approach", "playback"):
            for derivative in ("velocity", "acceleration", "jerk"):
                if not positive(config.get(phase, {}).get(part, {}).get(derivative)):
                    raise ValueError(f"Invalid {phase}/{part}/{derivative} planning bound")
    return config


def read_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text()))


def caps(config, phase, derivative):
    result = np.empty(54)
    for name, section in GROUPS.items():
        result[section] = config[phase]["arm" if name.endswith("arm") else "hand"][derivative]
    return result


def smooth_knots(raw, passes):
    """Convex averaging changes internal knots only; original endpoints retained."""
    smoothed = raw.copy()
    for _ in range(passes):
        smoothed[1:-1] = (smoothed[:-2] + 2 * smoothed[1:-1] + smoothed[2:]) / 4
    return smoothed


def rest_spline(times, positions):
    zeros = np.zeros(54)
    return make_interp_spline(times, positions, k=5, bc_type=([(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)]))


def extrema(spline):
    """All endpoint/stationary extrema for q, dq, ddq, dddq, not sampled estimates."""
    lower, upper = np.full((4, 54), np.inf), np.full((4, 54), -np.inf)
    start, end = spline.t[spline.k], spline.t[-spline.k - 1]
    for joint in range(54):
        polynomial = PPoly.from_spline((spline.t, spline.c[:, joint], spline.k))
        for order in range(4):
            curve = polynomial.derivative(order)
            roots = curve.derivative().roots(extrapolate=False)
            roots = roots[np.isfinite(roots) & (roots >= start) & (roots <= end)]
            knots = polynomial.x[(polynomial.x >= start) & (polynomial.x <= end)]
            values = curve(np.r_[knots, roots])
            lower[order, joint], upper[order, joint] = values.min(), values.max()
    return lower, upper


@dataclass
class Phase:
    name: str
    spline: object
    scale: float
    lower: np.ndarray
    upper: np.ndarray

    @property
    def duration(self):
        return float(self.spline.t[-self.spline.k - 1] * self.scale)

    def sample(self, seconds, derivative=0):
        seconds = np.asarray(seconds)
        if not np.isfinite(seconds).all() or np.any(seconds < 0) or np.any(seconds > self.duration + 1e-9):
            raise ValueError("Phase sample outside its lifetime")
        return (
            self.spline(np.clip(seconds / self.scale, 0, self.duration / self.scale), nu=derivative)
            / self.scale**derivative
        )


def make_phase(name, times, positions, config, limits, minimum_scale=1.0):
    spline = rest_spline(times, positions)
    low, high = extrema(spline)
    if np.any(low[0] < limits["lower"] - 1e-9) or np.any(high[0] > limits["upper"] + 1e-9):
        raise ValueError(f"{name}: continuous spline exceeds position limits; no clipping or automatic fallback")
    scale = float(minimum_scale)
    for order, derivative in enumerate(("velocity", "acceleration", "jerk"), 1):
        peak = np.maximum(np.abs(low[order]), np.abs(high[order]))
        scale = max(scale, float(np.max(peak / caps(config, name, derivative))) ** (1 / order))
    scale *= 1 + 1e-8
    return Phase(name, spline, scale, low, high)


@dataclass
class Plan:
    raw: np.ndarray
    knots: np.ndarray
    start: np.ndarray
    approach: Phase
    playback: Phase
    config: dict
    report: dict

    def arrays(self):
        arrays = {"raw_actions": self.raw, "smoothed_knots": self.knots, "recorded_start": self.start}
        for phase in (self.approach, self.playback):
            times = np.linspace(0, phase.duration, int(np.ceil(phase.duration * self.config["sample_hz"])) + 1)
            arrays[phase.name + "/time"] = times
            for order, key in enumerate(("position", "velocity", "acceleration", "jerk")):
                arrays[phase.name + "/" + key] = phase.sample(times, order)
        return arrays


def build_plan(raw, start, config, limits, names):
    validate_config(config)
    raw, start = checked_array(raw, (50, 54)), checked_array(start, (54,))
    for key in ("lower", "upper", "speed"):
        limits[key] = checked_array(limits[key], (54,))
    if np.any(limits["lower"] >= limits["upper"]) or np.any(limits["speed"] <= 0):
        raise ValueError("Invalid reference joint limits")
    if len(names) != 54 or len(set(names)) != 54:
        raise ValueError("Expected 54 unique joint names")
    for phase in ("approach", "playback"):
        if np.any(caps(config, phase, "velocity") > limits["speed"]):
            raise ValueError("Planning velocity exceeds existing reference limits")
    for value in (raw, start):
        if np.any(value < limits["lower"]) or np.any(value > limits["upper"]):
            raise ValueError("Raw actions or recorded start exceed reference joint limits")
    delta, knots = np.abs(raw[0] - start), smooth_knots(raw, config["smoothing_passes"])
    diagnostics = {}
    for group, section in GROUPS.items():
        part = "arm" if group.endswith("arm") else "hand"
        initial = float(delta[section].max())
        threshold = config[part + "_raw_initial_delta_rad"]
        if threshold is not None and initial > threshold:
            raise ValueError(f"{group}: raw initial delta {initial:.6f} exceeds {threshold}; reject before smoothing")
        modification = float(np.max(np.abs(knots[:, section] - raw[:, section])))
        if modification > config["max_smoothing_delta_rad"][part]:
            raise ValueError(f"{group}: smoothing changes the raw action too much")
        diagnostics[group] = {
            "raw_initial_delta_rad": initial,
            "raw_acquisition_limit_rad": threshold,
            "max_knot_modification_rad": modification,
            "raw_velocity_max_rad_s": float(np.abs(np.diff(raw[:, section], axis=0)).max() * 30),
            "raw_acceleration_max_rad_s2": float(np.abs(np.diff(raw[:, section], n=2, axis=0)).max() * 900),
        }
    # Two knots plus rest velocity/acceleration boundary conditions define the
    # quintic approach; all 54 joints share its duration.
    approach = make_phase("approach", np.array([0.0, 1.0]), np.stack([start, raw[0]]), config, limits)
    playback = make_phase("playback", np.arange(50) / 30, knots, config, limits, config["minimum_time_scale"])
    total = approach.duration + playback.duration + 2 * config["settle_seconds"]
    if total > config["max_total_seconds"]:
        raise ValueError(f"Required plan duration {total:.3f}s exceeds experiment limit; do not truncate")
    for phase in (approach, playback):
        for group, section in GROUPS.items():
            diagnostics[group][phase.name] = {
                derivative + "_max": float(
                    np.maximum(np.abs(phase.lower[order, section]), np.abs(phase.upper[order, section])).max()
                    / phase.scale**order
                )
                for order, derivative in enumerate(("velocity", "acceleration", "jerk"), 1)
            }
    report = {
        "schema_version": 1,
        "mode": "offline_weight_motion_evaluation",
        "hardware_output": False,
        "hardware_ready": False,
        "planning_checks_passed": True,
        "unverified": [
            "collision_path",
            "live_start_and_feedback",
            "hand_operating_limits_and_stop",
            "hardware_transport_and_final_deadline",
            "physical_tracking_and_stop",
        ],
        "source_hz": 30,
        "sample_hz": config["sample_hz"],
        "joint_names": names,
        "approach_seconds": approach.duration,
        "settle_seconds": config["settle_seconds"],
        "playback_seconds": playback.duration,
        "playback_time_scale": playback.scale,
        "requested_minimum_time_scale": config["minimum_time_scale"],
        "total_seconds": total,
        "groups": diagnostics,
        "reference_hashes": limits.get("sources", {}),
        "config": config,
        "explanation": "Approach is generated; playback uses smoothed predicted knots. Both start/end at rest. "
        "Analytic derivative bounds are offline planning checks, not evidence of safe hardware motion. "
        "A stored observation cannot authorize a live start.",
    }
    if config["hand_raw_initial_delta_rad"] is None:
        report["unverified"].append("hand_raw_acquisition_threshold")
    return Plan(raw, knots, start, approach, playback, config, report)


def load_record(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        if "state_at_return" not in data or data["state_at_return"].shape != (54,):
            raise ValueError("A 54-dimensional recorded state_at_return is required; no silent fallback")
        return (
            data["actions"].copy(),
            data["state_at_return"].copy(),
            {
                "record": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "start_basis": "historical measured state at inference return, NOT a live measurement",
            },
        )


def save_plan(plan, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "plan.npz", **plan.arrays())
    plan.report["artifact_sha256"] = hashlib.sha256((output / "plan.npz").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(plan.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
