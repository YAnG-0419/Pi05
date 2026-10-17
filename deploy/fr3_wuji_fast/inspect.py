"""Screen saved predictions at 30 Hz. No network, ROS, SDK or motion output."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from deploy.fr3_wuji_slow.core import ARM
from deploy.fr3_wuji_slow.core import HAND
from deploy.fr3_wuji_slow.core import Admission
from experiments.weight_motion_eval.reference import reference_limits

from .timeline import ARM_MOTION_LIMITS
from .timeline import arm_motion_settings
from .timeline import check_arm_speed
from .timeline import checked_chunk
from .timeline import limit_arm_speed


def inspect_record(path, limits, arm_motion_limits="disabled"):
    motion = arm_motion_settings(arm_motion_limits)
    row = {
        "record": str(path),
        "hardware_output": False,
        "input_kind": "historical_offline_only",
        "arm_motion_limits": motion["arm_motion_limits"],
        "arm_speed_enforced": motion["arm_speed_enforced"],
        "arm_speed_rad_s": motion["arm_speed_rad_s"],
    }
    with np.load(path, allow_pickle=False) as record:
        raw = record["actions"] if "actions" in record else record["raw_actions"]
    try:
        # Synthetic times exist only in this offline validator. Nothing is queued.
        chunk = checked_chunk(raw, Admission(1.0, 1.02, 1.1, "offline", "offline", 0), 1, limits)
        raw_speed = float(np.abs(np.diff(chunk.actions[:, ARM], axis=0)).max() * 30)
        if motion["arm_speed_enforced"]:
            actions, changed = limit_arm_speed(chunk.actions)
            check_arm_speed(actions)
        else:
            changed = 0
        row.update(
            arm_requested_speed_max_rad_s=raw_speed,
            arm_rate_limited_values=changed,
            hand_requested_speed_max_rad_s=float(np.abs(np.diff(chunk.actions[:, HAND], axis=0)).max() * 30),
            projected_hand_values=chunk.projected_hand_values,
        )
        row["status"] = "passes_with_arm_rate_limit" if changed else "passes_raw_chunk_checks"
    except ValueError as error:
        row.update(status="rejected", reason=str(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, action="append", required=True)
    parser.add_argument("--arm-motion-limits", choices=ARM_MOTION_LIMITS, default="disabled")
    args = parser.parse_args(argv)
    limits, _ = reference_limits()
    rows = [inspect_record(path, limits, args.arm_motion_limits) for path in args.record]
    print(
        json.dumps(
            {
                "hardware_output": False,
                "source_hz": 30,
                "arm_motion_limits": args.arm_motion_limits,
                "records": rows,
                "counts": dict(Counter(row["status"] for row in rows)),
                "note": "Does not validate live acquisition, RTG splice, physical tracking or stopping.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if any(row["status"] == "rejected" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
