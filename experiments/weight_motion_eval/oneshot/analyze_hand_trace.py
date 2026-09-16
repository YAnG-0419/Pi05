"""Compare SDK velocity with position differences using device timestamps."""

import argparse
from collections import deque
import json
import math
from pathlib import Path

from .hand import NIDS


def analyze(events):
    summary = []
    for index, nid in enumerate(NIDS):
        rows = []
        errors, flags = set(), set()
        for event in events:
            if event["event"] not in {"state", "diagnostics"}:
                continue
            for joint in event["joints"]:
                if joint["nid"] != nid:
                    continue
                if event["event"] == "state":
                    rows.append((event["device_timestamp_us"], joint["position"], joint["velocity"]))
                else:
                    if joint["error_code"]:
                        errors.add(joint["error_code"])
                    flags.update(
                        k
                        for k in ("position_limit_active", "velocity_limit_active", "current_limit_active")
                        if joint.get(k)
                    )
        peak_velocity = peak_diff = peak_window = 0.0
        above = duplicates = invalid = clock_reversals = 0
        previous = None
        window = deque()
        for stamp, q, velocity in rows:
            if not all(math.isfinite(v) for v in (stamp, q, velocity)):
                invalid += 1
                continue
            peak_velocity = max(peak_velocity, abs(velocity))
            above += abs(velocity) > math.radians(45)
            if previous is not None:
                dt = (stamp - previous[0]) / 1e6
                if dt == 0:
                    duplicates += 1
                    continue
                if dt < 0:
                    clock_reversals += 1
                    window.clear()
                elif dt <= 0.15:
                    peak_diff = max(peak_diff, abs(q - previous[1]) / dt)
            window.append((stamp, q))
            while len(window) >= 2 and stamp - window[1][0] >= 20000:
                window.popleft()
            dt = (stamp - window[0][0]) / 1e6
            if 0.02 <= dt <= 0.05:
                peak_window = max(peak_window, abs(q - window[0][1]) / dt)
            previous = (stamp, q)
        summary.append(
            {
                "joint_index": index,
                "nid": nid,
                "state_samples": len(rows),
                "sdk_peak_deg_s": math.degrees(peak_velocity),
                "position_difference_peak_deg_s": math.degrees(peak_diff),
                "position_20ms_peak_deg_s": math.degrees(peak_window),
                "samples_above_45_deg_s": above,
                "duplicate_timestamps": duplicates,
                "clock_reversals": clock_reversals,
                "nonfinite_samples": invalid,
                "firmware_errors": [f"0x{x:04X}" for x in sorted(errors)],
                "active_flags": sorted(flags),
            }
        )
    metadata = next((e for e in events if e["event"] == "metadata"), {})
    return {
        "metadata": metadata,
        "faults": [e for e in events if e["event"] == "fault_trigger"],
        "sent_commands": sum(e["event"] == "command_sent" for e in events),
        "joints": summary,
        "interpretation": "Position-derived speeds corroborate motion; neither differences nor a single SDK spike prove motor failure. 20 ms averages can hide shorter peaks. Timestamps are device values; clock reversals invalidate cross-boundary differences.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.trace.read_text().splitlines()]
    report = analyze(events)
    output = args.output or args.trace.with_suffix(".analysis.json")
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print("joint  SDK peak  position peak  20ms peak  [degrees/s]")
    for row in report["joints"]:
        print(
            f"{row['joint_index']:5d} {row['sdk_peak_deg_s']:9.2f} {row['position_difference_peak_deg_s']:14.2f} {row['position_20ms_peak_deg_s']:10.2f}"
        )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
