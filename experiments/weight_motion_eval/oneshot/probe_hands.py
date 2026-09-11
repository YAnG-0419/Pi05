"""Read-only hand identity, parameters and stopping-capability inventory.

Does not construct the teleoperation backend, publish commands, enable/disable,
clear faults, or alter parameters. Existing SDK owners cause a preflight failure.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    for owner in ("apps.operator_gui", "apps.wuji_ui", "adapters.wuji.hand_only", "observation_sources.py hands"):
        if owner in processes:
            raise RuntimeError("Existing hand owner: " + owner)
    import wuji_sdk

    result = {"hardware_output": False, "parameter_writes": False, "hands": {}}
    for side, address in (("left", "192.168.1.110:7447"), ("right", "192.168.2.111:7447")):
        hand = wuji_sdk.SdkManager.instance().connect(
            address=address,
            device_name="pi05_probe_" + side,
            options=wuji_sdk.ConnectOptions(enable_bridge=False, auto_time_sync_interval_ms=None),
        )
        sub = None
        try:
            if str(hand.handedness().get()).lower() != side or hand.online_joints_count().get() != 20:
                raise RuntimeError("Hand identity or online count mismatch")
            entry = {
                "address": address,
                "serial": hand.serial_number,
                "info": repr(hand.info),
                "hardware_version": repr(hand.hw_version().get()),
                "mit_gains": [
                    {"kp": float(p.kp), "kd": float(p.kd)} if p is not None else None for p in hand.mit_params().get()
                ],
                "current_limits_a": hand.effort_limit().get(),
                "stop_related_resources": [],
            }
            for resource in hand.resources():
                description = str(resource)
                if any(word in description.lower() for word in ("watchdog", "timeout", "emergency", "stop", "limit")):
                    entry["stop_related_resources"].append(description)
            sub = hand.joint_diagnostics().subscribe()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                frame = sub.recv()
                if frame is not None and len(frame.joints) == 20:
                    entry["diagnostics"] = [
                        {
                            "nid": int(j.nid),
                            "state": int(j.status_word.ext_state),
                            "error": int(j.error_code_current),
                            "current_a": float(j.current),
                        }
                        for j in frame.joints
                    ]
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("No full diagnostic frame")
            result["hands"][side] = entry
        finally:
            if sub is not None:
                sub.close()
            hand.disconnect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
