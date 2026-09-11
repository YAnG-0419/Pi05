"""Independent weight-motion evaluation: capture, plan, preview and simulate."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def safe_output(value):
    output = Path(value).resolve()
    base = (ROOT / "logs/weight_motion_eval").resolve()
    if not output.is_relative_to(base) or output == base:
        raise ValueError(f"Output must be a new child directory of {base}")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    return output


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np

    from experiments.weight_motion_eval.planner import build_plan
    from experiments.weight_motion_eval.planner import load_record
    from experiments.weight_motion_eval.planner import read_config
    from experiments.weight_motion_eval.planner import save_plan
    from experiments.weight_motion_eval.planner import validate_config
    from experiments.weight_motion_eval.preview import write_preview
    from experiments.weight_motion_eval.reference import reference_limits
    from experiments.weight_motion_eval.rehearsal import run_rehearsal

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "rehearse", "oneshot-simulate"):
        command = commands.add_parser(name)
        command.add_argument("--record", type=Path, required=True, help="Existing inference-NNNN.npz")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--time-scale", type=float, help="Minimum slowdown; bounds can require a larger value")
        if name == "rehearse":
            command.add_argument("--inject", choices=("feedback_loss", "pause", "scheduler_delay", "tracking_error"))
        if name == "oneshot-simulate":
            command.add_argument(
                "--inject", choices=("feedback_loss", "consumer_delay", "scheduler_delay", "pause", "partial_submit")
            )
    batch = commands.add_parser("batch", help="Check a directory of recorded chunks without hardware")
    batch.add_argument("--records", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--time-scale", type=float)
    capture = commands.add_parser("capture", help="Read ROS arm feedback and infer once; never moves hardware")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--uri", default="ws://127.0.0.1:8000")
    capture.add_argument("--start-cameras", action="store_true")
    capture.add_argument("--hand-source", choices=("ros", "sdk"), default="ros")
    args = parser.parse_args()
    config = read_config(args.config)
    output = safe_output(args.output)
    if args.command == "capture":
        from experiments.weight_motion_eval.capture import capture_once

        capture_once(output, uri=args.uri, start_cameras=args.start_cameras, hand_source=args.hand_source)
        print(f"Captured: {output}")
        return
    if args.time_scale is not None:
        config["minimum_time_scale"] = args.time_scale
    validate_config(config)
    limits, names = reference_limits()
    if args.command == "batch":
        records = sorted(args.records.glob("inference-*.npz"))
        if not records:
            raise ValueError("No inference-*.npz files in the source directory")
        output.mkdir(parents=True, exist_ok=False)
        accepted, rejected = [], []
        for record in records:
            try:
                raw, start, source = load_record(record)
                plan = build_plan(raw, start, config, limits, names)
                accepted.append(
                    {
                        "record": str(record),
                        "sha256": source["sha256"],
                        "approach_seconds": plan.approach.duration,
                        "playback_seconds": plan.playback.duration,
                        "time_scale": plan.playback.scale,
                        "groups": plan.report["groups"],
                    }
                )
            except ValueError as error:
                rejected.append({"record": str(record), "reason": str(error)})
        result = {
            "mode": "offline_batch",
            "hardware_output": False,
            "hardware_ready": False,
            "records": len(records),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejection_reasons": dict(Counter(row["reason"].split(":")[0] for row in rejected)),
            "config": config,
            "reference_hashes": limits.get("sources", {}),
            "accepted": accepted,
            "rejected": rejected,
        }
        (output / "batch-report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {key: result[key] for key in ("records", "accepted_count", "rejected_count", "rejection_reasons")}
            )
        )
        return
    raw, start, source = load_record(args.record)
    plan = build_plan(raw, start, config, limits, names)
    plan.report["source"] = source
    save_plan(plan, output)
    write_preview(plan, output / "preview.html")
    if args.command == "oneshot-simulate":
        from experiments.weight_motion_eval.oneshot.runner import simulate

        report, arrays = simulate(plan, limits, inject=args.inject)
        report["source"] = source
        (output / "oneshot.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(output / "oneshot.npz", **arrays)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        expected = "complete" if args.inject is None else "stopped" if args.inject == "pause" else "fault"
        if report["state"] != expected:
            raise RuntimeError(f"Single-shot simulation expected {expected}; inspect oneshot.json")
    if args.command == "rehearse":
        report, arrays = run_rehearsal(plan, inject=args.inject)
        (output / "rehearsal.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        np.savez_compressed(output / "rehearsal.npz", **arrays)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.inject is None and report["state"] != "complete":
            raise RuntimeError("Virtual rehearsal failed; inspect rehearsal.json")
        if args.inject is not None and (not report["injection_applied"] or report["state"] not in {"fault", "paused"}):
            raise RuntimeError("Fault injection did not stop rehearsal")
    print(
        json.dumps(
            {
                "hardware_output": False,
                "approach_seconds": plan.approach.duration,
                "playback_seconds": plan.playback.duration,
                "time_scale": plan.playback.scale,
                "preview": str(output / "preview.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
