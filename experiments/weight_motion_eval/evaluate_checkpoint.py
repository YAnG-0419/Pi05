"""Sample historical real observations for offline checkpoint diagnostics; no device I/O."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    from experiments.weight_motion_eval.planner import GROUPS

    result = {"count": len(rows), "groups": {}}
    for group in GROUPS:
        values = [row["groups"][group] for row in rows]
        result["groups"][group] = {
            "range_violation_chunks": sum(v["range_violation_values"] > 0 for v in values),
            "range_violation_values": sum(v["range_violation_values"] for v in values),
            "range_violation_joints": sorted({j for v in values for j in v["range_violation_joints"]}),
            "range_excess_max_rad": max(v["range_excess_max_rad"] for v in values),
            "speed_exceedance_chunks": sum(v["nominal_speed_exceedance_values"] > 0 for v in values),
        }
        for metric in (
            "first_delta_aligned_state_max_rad",
            "first_delta_return_state_max_rad",
            "nominal_velocity_max_rad_s",
            "step_delta_max_rad",
        ):
            result["groups"][group][metric] = dict(
                zip(
                    ("min", "median", "p95", "max"),
                    map(float, np.percentile([v[metric] for v in values], [0, 50, 95, 100])),
                    strict=True,
                )
            )
        if group.endswith("arm"):
            result["groups"][group]["initial_delta_exceeds_gateway_chunks"] = sum(
                v["initial_delta_exceeds_gateway_at_return"] for v in values
            )
    result["range_violation_chunks_any"] = sum(
        any(v["range_violation_values"] > 0 for v in row["groups"].values()) for row in rows
    )
    result["initial_arm_delta_exceeds_gateway_chunks_any"] = sum(
        any(row["groups"][g]["initial_delta_exceeds_gateway_at_return"] for g in ("left_arm", "right_arm"))
        for row in rows
    )
    result["speed_exceedance_chunks_any"] = sum(
        any(v["nominal_speed_exceedance_values"] > 0 for v in row["groups"].values()) for row in rows
    )
    return result


def main():
    sys.path.insert(0, str(ROOT))
    from experiments.weight_motion_eval.planner import GROUPS
    from experiments.weight_motion_eval.planner import build_plan
    from experiments.weight_motion_eval.planner import read_config
    from experiments.weight_motion_eval.planner import save_plan
    from experiments.weight_motion_eval.preview import write_preview
    from experiments.weight_motion_eval.reference import deployment_module
    from experiments.weight_motion_eval.reference import reference_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-session", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.per_session < 1 or args.repeats < 1:
        parser.error("sample counts must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "logs/weight_motion_eval"):
        parser.error("output must be within logs/weight_motion_eval")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    stats_files = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(stats_files) != 1:
        raise ValueError("Expected exactly one checkpoint-owned normalization file")
    stats_path = stats_files[0]
    stats_json = json.loads(stats_path.read_text())["norm_stats"]
    for group in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            a = np.asarray(stats_json[group][field])
            if a.shape != (54,) or not np.isfinite(a).all():
                raise ValueError("Invalid normalization statistics")
        if np.any(np.asarray(stats_json[group]["std"]) < 0) or np.any(
            np.asarray(stats_json[group]["q01"]) > stats_json[group]["q99"]
        ):
            raise ValueError("Invalid normalization ranges")
    rng = np.random.default_rng(args.seed)
    sessions = sorted((ROOT / "logs/fr3_wuji").glob("inference-*/"))
    selected = []
    for session in sessions:
        files = sorted(session.glob("inference-*.npz"))
        if files:
            selected.extend(
                files[int(i)]
                for i in sorted(rng.choice(len(files), size=min(args.per_session, len(files)), replace=False))
            )
    if not selected:
        raise ValueError("No recorded real observations")
    limits, names = reference_limits()
    diagnostics = deployment_module("record_inference")
    config = read_config(Path(__file__).with_name("config.yaml"))
    report = {
        "mode": "offline_checkpoint_inference_on_historical_real_observations",
        "hardware_output": False,
        "hardware_ready": False,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "norm_stats": {"path": str(stats_path), "sha256": digest(stats_path)},
        "parameter_manifest_sha256": digest(checkpoint / "params/manifest.ocdbt"),
        "parameter_metadata_sha256": digest(checkpoint / "params/_METADATA"),
        "training_config_assumption": "pi05_fr3_wuji: arm delta actions, absolute hand actions, 30 Hz, 50x54; normalization loaded explicitly from this checkpoint",
        "source_counts": dict(Counter(str(p.parent) for p in selected)),
        "source_files": [{"path": str(p), "sha256": digest(p)} for p in selected],
        "reference_hashes": limits["sources"],
        "planner_config": config,
        "limitations": [
            "Historical states are not live feedback",
            "Three short sessions, not task-wide coverage",
            "No collision, contact force or task success evaluation",
            "Hand operating parameters remain unverified",
            "Old baseline consists of saved outputs, not matched-noise reruns",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected {len(selected)} observations; {args.repeats} noise draws each; stats: {stats_path}", flush=True)
    import jax

    from openpi.policies import policy_config
    from openpi.training import checkpoints
    from openpi.training import config as training_config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU required")
    asset_id = str(stats_path.parent.relative_to(checkpoint / "assets"))
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", asset_id)
    started = time.monotonic()
    policy = policy_config.create_trained_policy(
        training_config.get_config("pi05_fr3_wuji"), checkpoint, norm_stats=norm_stats
    )
    report["load_seconds"] = time.monotonic() - started
    print(f"Loaded in {report['load_seconds']:.2f}s; warming up", flush=True)
    rows, old_rows, variations, accepted = [], [], [], []
    for sample_index, source in enumerate(selected):
        with np.load(source, allow_pickle=False) as z:
            obs = {key: z[key].copy() for key in z.files if key.startswith("observation/")}
            obs["prompt"] = str(z["prompt"].item())
            at_send, at_return = z["state_at_send"].copy(), z["state_at_return"].copy()
            old_actions = z["actions"].copy()
        if at_return.shape != (54,):
            raise ValueError(f"Missing historical state: {source}")
        old_rows.append(
            {
                "source": str(source),
                "groups": diagnostics.analyze(old_actions, obs["observation/state"], at_send, at_return, limits),
            }
        )
        if sample_index == 0:
            start = time.monotonic()
            policy.infer(obs, noise=np.zeros((50, 54), dtype=np.float32))
            report["warmup_seconds"] = time.monotonic() - start
            print(f"Warmup completed in {report['warmup_seconds']:.2f}s", flush=True)
        draws = []
        for repeat in range(args.repeats):
            noise = rng.standard_normal((50, 54)).astype(np.float32)
            start = time.monotonic()
            result = policy.infer(obs, noise=noise)
            actions = diagnostics.validate_actions(result["actions"])
            elapsed = time.monotonic() - start
            draws.append(actions)
            index = len(rows)
            np.savez_compressed(
                output / f"inference-{index:04d}.npz",
                actions=actions,
                state_at_return=at_return,
                state_at_send=at_send,
                observed_state=obs["observation/state"],
                noise=noise,
                source=str(source),
            )
            row = {
                "index": index,
                "source": str(source),
                "repeat": repeat,
                "inference_seconds": elapsed,
                "groups": diagnostics.analyze(actions, obs["observation/state"], at_send, at_return, limits),
            }
            try:
                plan = build_plan(actions, at_return, config, limits, names)
                row["planning"] = {
                    "accepted": True,
                    "time_scale": plan.playback.scale,
                    "approach_seconds": plan.approach.duration,
                    "playback_seconds": plan.playback.duration,
                }
                accepted.append(row)
            except ValueError as error:
                row["planning"] = {"accepted": False, "reason": str(error)}
            rows.append(row)
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        variations.append(
            {
                "source": str(source),
                "groups": {
                    group: {
                        "max_first_step_spread_rad": float(np.ptp(np.stack(draws)[:, 0, section], axis=0).max()),
                        "max_horizon_spread_rad": float(np.ptp(np.stack(draws)[:, :, section], axis=0).max()),
                    }
                    for group, section in GROUPS.items()
                },
            }
        )
        print(
            f"Observation {sample_index + 1}/{len(selected)}, predictions {len(rows)}, planner accepted {len(accepted)}",
            flush=True,
        )
    report["new_summary"], report["old_recorded_summary"] = summarize(rows), summarize(old_rows)
    report["inference_seconds"] = dict(
        zip(
            ("min", "median", "p95", "max"),
            map(float, np.percentile([row["inference_seconds"] for row in rows], [0, 50, 95, 100])),
            strict=True,
        )
    )
    report["planning"] = {
        "accepted_count": len(accepted),
        "rejected_count": len(rows) - len(accepted),
        "reasons": dict(
            Counter(row["planning"]["reason"].split(":")[0] for row in rows if not row["planning"]["accepted"])
        ),
    }
    if accepted:
        for key in ("time_scale", "approach_seconds", "playback_seconds"):
            report["planning"][key] = dict(
                zip(
                    ("min", "median", "max"),
                    map(float, np.percentile([row["planning"][key] for row in accepted], [0, 50, 100])),
                    strict=True,
                )
            )
        representative = max(accepted, key=lambda row: row["planning"]["time_scale"])
        with np.load(output / f"inference-{representative['index']:04d}.npz") as z:
            plan = build_plan(z["actions"], z["state_at_return"], config, limits, names)
        plan.report["evaluation_index"] = representative["index"]
        save_plan(plan, output / "slowest_accepted")
        write_preview(plan, output / "slowest_accepted/preview.html")
    report["repeat_variability"] = variations
    report["old_recorded_results"] = old_rows
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("new_summary", "planning", "inference_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
