"""Read hand identity, online count and faults without motor/parameter writes."""

import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="192.168.2.111:7447")
    args = parser.parse_args()
    # Fail clearly before SDK initialization when the execution environment
    # forbids network sockets. This probe sends no packets.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM):
            pass
    except PermissionError as error:
        raise SystemExit("Network access blocked by execution environment; no device queried: " + str(error)) from error

    import wuji_sdk as sdk

    hand = None
    subscription = None
    result = {"mode": "read_only", "address": args.address}
    try:
        hand = sdk.SdkManager.instance().connect(
            address=args.address,
            device_name="pi05_readonly_hand_status",
            options=sdk.ConnectOptions(enable_bridge=False, auto_time_sync_interval_ms=None),
        )
        result.update(serial=str(hand.serial_number), handedness=str(hand.handedness().get()))
        result["online_joints_count"] = int(hand.online_joints_count().get())
        subscription = hand.joint_diagnostics().subscribe()
        deadline = time.monotonic() + 3
        newest = None
        while time.monotonic() < deadline:
            for _ in range(256):
                frame = subscription.recv()
                if frame is None:
                    break
                newest = frame
            if newest is not None:
                break
            time.sleep(0.01)
        if newest is None:
            raise RuntimeError("No diagnostic frame received within 3 seconds")
        result["diagnostic_age_seconds"] = time.time() - int(newest.header.timestamp_us) / 1e6
        result["diagnostic_joint_count"] = len(newest.joints)
        result["joints"] = [
            {
                "nid": int(joint.nid),
                "state": int(joint.status_word.ext_state),
                "state_name": str(joint.status_word.ext_state_name),
                "error_code": int(joint.error_code_current),
                "error_hex": f"0x{int(joint.error_code_current):04X}",
                "error_description": sdk.WujiHand2.describe_error(int(joint.error_code_current)),
            }
            for joint in sorted(newest.joints, key=lambda joint: int(joint.nid))
        ]
        expected = {finger * 5 + joint + 1 for finger in range(5) for joint in range(4)}
        result["missing_joint_ids"] = sorted(expected - {int(joint.nid) for joint in newest.joints})
    except Exception as error:
        result["read_error"] = str(error)
        raise
    finally:
        try:
            if subscription is not None:
                subscription.close()
        finally:
            if hand is not None:
                hand.disconnect()
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
