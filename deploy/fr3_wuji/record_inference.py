"""Infer from live observations and save diagnostics. No hardware command interface."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from observation import STATE_ORDER
from observation import ObservationUnavailableError
from observation import joint_names
import observe
from openpi_client import msgpack_numpy
from websockets.sync.client import connect
import yaml

NAMES = tuple(name for source in STATE_ORDER for name in joint_names(source))
GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 27), "right_arm": slice(27, 34), "right_hand": slice(34, 54)}


def load_limits(reference=observe.REFERENCE):
    """Read existing configured limits without importing hardware/control code."""
    safety = reference / "ros_ws/src/teleop_core/teleop_core/safety.py"
    assignments = {}
    for node in ast.parse(safety.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("LOWER_LIMITS", "UPPER_LIMITS"):
                    expr = node.value.args[0]
                    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult)):
                        raise ValueError("Reference arm limit format changed")
                    assignments[target.id] = np.asarray(ast.literal_eval(expr.left) * ast.literal_eval(expr.right))
    if any(assignments.get(key, np.empty(0)).shape != (14,) for key in ("LOWER_LIMITS", "UPPER_LIMITS")):
        raise ValueError("Expected 14 named arm limits")
    mode = reference / "config/modes/teleop_control.yaml"
    config = yaml.safe_load(mode.read_text())["teleop_safety_gateway"]["ros__parameters"]
    lower, upper, speed = np.empty(54), np.empty(54), np.empty(54)
    paths = [safety, mode]
    for side, offset in (("left", 0), ("right", 7)):
        arm = GROUPS[side + "_arm"]
        lower[arm] = assignments["LOWER_LIMITS"][offset : offset + 7]
        upper[arm] = assignments["UPPER_LIMITS"][offset : offset + 7]
        speed[arm] = config["max_joint_speed"]
        hand_config = reference / f"adapters/wuji/config/retarget_manus_wuji_hand_2_{side}.yaml"
        urdf = (hand_config.parent / yaml.safe_load(hand_config.read_text())["optimizer"]["urdf_path"]).resolve()
        paths.extend((hand_config, urdf))
        joints = {node.attrib["name"]: node.find("limit") for node in ET.parse(urdf).getroot().findall("joint")}
        for index, name in enumerate(joint_names(side + "_hand"), GROUPS[side + "_hand"].start):
            limit = joints[name]
            lower[index], upper[index], speed[index] = [float(limit.attrib[k]) for k in ("lower", "upper", "velocity")]
    if not (np.isfinite([lower, upper, speed]).all() and np.all(lower < upper) and np.all(speed > 0)):
        raise ValueError("Invalid reference limits")
    return {
        "lower": lower,
        "upper": upper,
        "speed": speed,
        "arm_initial_delta": float(config["max_initial_delta"]),
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def latest_state(store, *, now_wall=None, now_mono=None):
    wall = time.time() if now_wall is None else now_wall
    mono = time.monotonic() if now_mono is None else now_mono
    with store.lock:
        if store.clock_offset is None or abs(wall - mono - store.clock_offset) > 0.05:
            raise ObservationUnavailableError("clock_jump")
        values, ages = [], {}
        for source in STATE_ORDER:
            history = store.buffers[source]
            if not history:
                raise ObservationUnavailableError("missing:" + source)
            sample = history[-1]
            age, receive_age = wall - sample.stamp, mono - sample.received
            if not -0.02 <= age <= store.state_live_age or not 0 <= receive_age <= store.state_live_age:
                raise ObservationUnavailableError("stale:" + source)
            values.append(sample.value)
            ages[source] = {"source_age_seconds": age, "receive_age_seconds": receive_age}
        return np.concatenate(values), ages


def input_ages(metadata, wall):
    return {key: wall - sample["stamp"] for key, sample in metadata["samples"].items()}


def validate_actions(actions):
    actions = np.asarray(actions)
    if actions.shape != (50, 54) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite (50, 54) actions, got {actions.shape}")
    return actions


def analyze(actions, observed, at_send, at_return, limits, hz=30):
    actions = validate_actions(actions)
    excess = np.maximum(np.maximum(limits["lower"] - actions, actions - limits["upper"]), 0)
    velocity = np.abs(np.diff(actions, axis=0)) * hz
    groups = {}
    for name, section in GROUPS.items():
        indices = np.arange(54)[section]
        first = np.abs(actions[0, section] - observed[section])
        violation = excess[:, section] > 0
        group = {
            "first_delta_aligned_state_max_rad": float(first.max()),
            "first_delta_send_state_max_rad": float(np.max(np.abs(actions[0, section] - at_send[section]))),
            "first_delta_return_state_max_rad": None
            if at_return is None
            else float(np.max(np.abs(actions[0, section] - at_return[section]))),
            "horizon_delta_aligned_state_max_rad": float(np.max(np.abs(actions[:, section] - observed[section]))),
            "range_violation_values": int(violation.sum()),
            "range_violation_joints": [NAMES[i] for i in indices[violation.any(axis=0)]],
            "range_excess_max_rad": float(excess[:, section].max()),
            "step_delta_max_rad": float(np.abs(np.diff(actions[:, section], axis=0)).max()),
            "nominal_velocity_max_rad_s": float(velocity[:, section].max()),
            "nominal_speed_exceedance_values": int((velocity[:, section] > limits["speed"][section]).sum()),
        }
        if name.endswith("arm"):
            group["initial_delta_exceeds_gateway_at_send"] = (
                group["first_delta_send_state_max_rad"] > limits["arm_initial_delta"]
            )
            group["initial_delta_exceeds_gateway_at_return"] = (
                None if at_return is None else (group["first_delta_return_state_max_rad"] > limits["arm_initial_delta"])
            )
        groups[name] = group
    return groups


class Recorder:
    def __init__(self, ws, limits, *, timeout=20, max_input_age=None):
        self.ws, self.limits, self.timeout = ws, limits, timeout
        if max_input_age is not None and (not np.isfinite(max_input_age) or max_input_age <= 0):
            raise ValueError("Input age budget must be positive and finite")
        self.max_input_age = max_input_age
        self.packer = msgpack_numpy.Packer()
        self.runs, self.rejected = [], Counter()
        self.actions = []
        self.completed = 0

    def __call__(self, store, obs, metadata, output):
        # Pack before the final freshness check; image serialization takes time.
        pack_started = time.monotonic()
        request = self.packer.pack(obs)
        send_wall, send_mono = time.time(), time.monotonic()
        ages = input_ages(metadata, send_wall)
        if any(not -0.02 <= age <= store.max_age for age in ages.values()):
            self.rejected["input_expired_before_send"] += 1
            return
        if self.max_input_age is not None and max(ages.values()) > self.max_input_age:
            self.rejected["input_exceeds_inference_age_budget"] += 1
            return
        try:
            send_state, send_state_ages = latest_state(store, now_wall=send_wall, now_mono=send_mono)
        except ObservationUnavailableError as error:
            self.rejected[str(error)] += 1
            return
        self.ws.send(request)
        payload = self.ws.recv(timeout=self.timeout)
        returned_mono, returned_wall = time.monotonic(), time.time()
        if isinstance(payload, str):
            raise RuntimeError(f"Inference server error: {payload}")
        result = msgpack_numpy.unpackb(payload)
        actions = validate_actions(result["actions"])
        elapsed = returned_mono - send_mono
        clock_jump = abs((returned_wall - send_wall) - elapsed) > 0.05
        # Advance original ages by monotonic latency, so a backwards wall-clock
        # jump cannot make the input look fresh again.
        return_ages = {key: age + elapsed for key, age in ages.items()}
        return_state, return_state_ages, return_error = None, None, None
        try:
            return_state, return_state_ages = latest_state(store)
        except ObservationUnavailableError as error:
            return_error = str(error)
        run = {
            "index": self.completed,
            "send_wall": send_wall,
            "return_wall": returned_wall,
            "roundtrip_seconds": elapsed,
            "serialization_seconds": send_mono - pack_started,
            "input": metadata,
            "input_age_at_send_seconds": ages,
            "input_age_at_return_seconds": return_ages,
            "input_expired_at_return": clock_jump or max(return_ages.values()) > store.max_age,
            "clock_jump_during_inference": clock_jump,
            "latest_state_at_send": send_state_ages,
            "latest_state_at_return": return_state_ages,
            "return_state_error": return_error,
            "server_timing": result.get("server_timing", {}),
            "policy_timing": result.get("policy_timing", {}),
            "groups": analyze(actions, obs["observation/state"], send_state, return_state, self.limits),
        }
        save_started = time.monotonic()
        np.savez_compressed(
            output / f"inference-{self.completed:04d}.npz",
            **obs,
            actions=actions,
            state_at_send=send_state,
            state_at_return=np.empty(0) if return_state is None else return_state,
        )
        run["artifact_save_seconds"] = time.monotonic() - save_started
        with (output / "inferences.jsonl").open("a") as stream:
            stream.write(json.dumps(run, allow_nan=False) + "\n")
        self.runs.append(run)
        self.actions.append(actions)
        self.completed += 1
        if self.completed == 1 or self.completed % 10 == 0:
            print(
                f"Recorded {self.completed}: {elapsed * 1000:.1f} ms, "
                f"input age at return {max(return_ages.values()) * 1000:.1f} ms",
                flush=True,
            )

    def finish(self, output, metadata, error=None):
        report = {
            "hardware_output": False,
            "status": "recorded" if self.completed >= 10 and error is None else "incomplete",
            "error": error,
            "server_metadata": metadata,
            "completed": self.completed,
            "rejected_before_inference": dict(self.rejected),
            "max_input_age_at_send_seconds": self.max_input_age,
            "joint_names": NAMES,
            "units": "radian",
            "nominal_action_hz": 30,
            "limits": {
                key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in self.limits.items()
            },
            "interpretation": "Raw absolute targets; no clipping or delta addition. Arms use existing gateway "
            "limits and 0.05 rad acquisition threshold; hands use configured Hand 2 URDF position/velocity "
            "limits, not firmware readback. Hand acquisition threshold is not established. "
            "Velocity is a 30 Hz finite-difference diagnostic, not executed motion. "
            "No collision, torque or closed-loop validation. This is not execution approval.",
        }
        if self.runs:
            for key, values in {
                "roundtrip_ms": [r["roundtrip_seconds"] * 1000 for r in self.runs],
                "input_age_send_ms": [max(r["input_age_at_send_seconds"].values()) * 1000 for r in self.runs],
                "input_age_return_ms": [max(r["input_age_at_return_seconds"].values()) * 1000 for r in self.runs],
                "serialization_ms": [r["serialization_seconds"] * 1000 for r in self.runs],
                "artifact_save_ms": [r["artifact_save_seconds"] * 1000 for r in self.runs],
            }.items():
                report[key] = dict(
                    zip(("min", "median", "p95", "max"), np.percentile(values, [0, 50, 95, 100]).tolist(), strict=True)
                )
            report["expired_at_return_count"] = sum(r["input_expired_at_return"] for r in self.runs)
            report["return_state_unavailable_count"] = sum(r["return_state_error"] is not None for r in self.runs)
            report["groups"] = {}
            for name in GROUPS:
                groups = [r["groups"][name] for r in self.runs]
                report["groups"][name] = {
                    key: max(g[key] for g in groups if g[key] is not None)
                    for key in groups[0]
                    if "max" in key and any(g[key] is not None for g in groups)
                }
                report["groups"][name].update(
                    range_violation_values=sum(g["range_violation_values"] for g in groups),
                    range_violation_joints=sorted({j for g in groups for j in g["range_violation_joints"]}),
                    nominal_speed_exceedance_values=sum(g["nominal_speed_exceedance_values"] for g in groups),
                )
                if name.endswith("arm"):
                    report["groups"][name]["initial_delta_exceeds_gateway_at_send_count"] = sum(
                        g["initial_delta_exceeds_gateway_at_send"] for g in groups
                    )
            all_actions = np.concatenate(self.actions)
            with (output / "joint-ranges.csv").open("w") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["joint", "lower_rad", "upper_rad", "output_min_rad", "output_max_rad", "out_of_range_values"]
                )
                for i, name in enumerate(NAMES):
                    low, high = self.limits["lower"][i], self.limits["upper"][i]
                    values = all_actions[:, i]
                    writer.writerow(
                        [name, low, high, values.min(), values.max(), int(((values < low) | (values > high)).sum())]
                    )
        (output / "inference-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument(
        "--max-input-age",
        type=float,
        default=0.07,
        help="Maximum input age at send; reserve latency within the unchanged 200 ms observation limit.",
    )
    parser.add_argument("--camera-profile", choices=("recording", "rgb"), default=None)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args, observation_args = parser.parse_known_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not np.isfinite(args.max_input_age) or not 0 < args.max_input_age <= 0.2:
        parser.error("--max-input-age must be positive and at most 0.2 seconds")
    if args.output.exists():
        parser.error("Use a new output directory")
    limits = load_limits()
    with connect(args.uri, compression=None, max_size=None, open_timeout=10, close_timeout=5, proxy=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=args.timeout))
        expected = {
            "config": "pi05_fr3_wuji",
            "action_dim": 54,
            "action_horizon": 50,
            "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
            "hardware_output": False,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Unexpected inference contract: {metadata}")
        recorder = Recorder(ws, limits, timeout=args.timeout, max_input_age=args.max_input_age)
        if args.camera_profile is not None or "--start-cameras" in observation_args:
            observation_args += ["--camera-profile", args.camera_profile or "rgb"]
        observation_args += ["--poll-interval", str(args.poll_interval)]
        error = None
        try:
            observe.main([*observation_args, "--output", str(args.output)], consumer=recorder)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if args.output.exists():
                report = recorder.finish(args.output, metadata, error)
                print(f"Inference report: {args.output / 'inference-report.json'}", flush=True)
        if report["status"] != "recorded":
            raise RuntimeError("Insufficient completed real inferences")


if __name__ == "__main__":
    main()
