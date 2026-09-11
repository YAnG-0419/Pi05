"""Single-owner ROS/SDK process. Default read-only; no model dependencies."""

import argparse
import json
import os
from pathlib import Path
import select
import signal
import socket
import struct
import sys
import time

from .devices import DeviceSession
from .hand import HandOwner
from .ipc import encode
from .ipc import receive
from .ipc import unpack_frame
from .ipc import wire_feedback
from .qualification import Qualification
from .qualification import SupervisedTrial
from .qualification import digest


def serve_connection(conn, session, *, stopping, pump):
    conn.settimeout(0.002)
    last_request = 0
    # The client allows one in-flight RPC. EOF or another packet while enable
    # is pending means cancellation, not permission to continue acquisition.
    session.cancel_requested = lambda: stopping() or bool(select.select([conn], [], [], 0)[0])
    while not stopping():
        pump()
        session.service()
        try:
            request = receive(conn)
        except TimeoutError:
            continue
        except EOFError:
            session.stop()
            return
        request_id = request.get("id")
        try:
            if type(request_id) is not int or request_id != last_request + 1:
                raise ValueError("Out-of-order IPC request")
            last_request = request_id
            operation = request["operation"]
            if operation == "inventory":
                if session.state != "readonly":
                    raise ValueError("Readback of operating parameters is only allowed before acquisition")
                result = {side: hand.identity() for side, hand in session.hands.items()}
            elif operation == "readiness":
                if session.state != "readonly":
                    raise ValueError("Readiness snapshot is only available before acquisition")
                result = {
                    "hands": {
                        side: {
                            "state_received": hand.latest is not None,
                            "error": session.arms.hand_feedback_errors.get(side),
                        }
                        for side, hand in session.hands.items()
                    },
                    "arm_state_sides": sorted(session.arms.samples),
                }
            elif operation == "feedback":
                result = wire_feedback(session.feedback())
            elif operation == "prepare":
                session.prepare(request["run_id"], request["plan_hash"], request["start"], request["finish_policy"])
                result = {"state": session.state}
            elif operation == "submit":
                result = session.submit(unpack_frame(request["frame"]))
            elif operation == "finish":
                result = session.finish()
            elif operation == "monitor":
                if session.state != "holding":
                    raise RuntimeError("Hand hold monitor lost ownership: " + str(session.fault))
                session.feedback()
                session.last_activity = session.clock()
                result = {"state": session.state}
            elif operation == "stop":
                result = session.stop()
            else:
                raise ValueError("Unknown device operation")
            reply = {"id": request_id, "ok": True, "result": result}
        except Exception as error:
            # Missing feedback is expected during read-only startup. Any error
            # after acquisition is terminal, including failed finish/monitor.
            if session.state != "readonly":
                session.fault = str(error)
                session.stop()
            reply = {"id": request_id, "ok": False, "error": str(error)}
        conn.sendall(encode(reply))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--supervised-trial", action="store_true")
    args = parser.parse_args()
    if args.supervised_trial and (not args.execute or args.qualification is not None):
        parser.error("Supervised trial must explicitly enable output and cannot claim commissioning")
    config = json.loads(args.runtime.read_text())
    for path, expected in config["reference_hashes"].items():
        if digest(path) != expected:
            raise ValueError("Runtime reference limits changed: " + path)
    qualification = None
    if args.execute:
        if digest(config["controller_library"]) != config["controller_sha256"]:
            raise ValueError("Controller artifact changed")
        qualification = (
            SupervisedTrial()
            if args.supervised_trial
            else Qualification(args.qualification, controller_sha256=config["controller_sha256"])
        )
    sys.path.insert(0, config["reference"] + "/ros_ws/src/teleop_core")
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    import wuji_sdk

    from .ros_devices import RosArms

    stopping = False

    def interrupt(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    hands, arms, session = {}, None, None
    try:
        arms = RosArms(publish_reset_idle=args.execute, output_enabled=args.execute)
        for side in ("left", "right"):
            hands[side] = HandOwner(wuji_sdk, side, config["hands"][side])
        if isinstance(qualification, SupervisedTrial):
            qualification.bind_hands({side: hand.identity() for side, hand in hands.items()})
            (args.runtime.parent / "trial-device-readback.json").write_text(
                json.dumps(
                    {
                        **qualification.data,
                        "hands": qualification.hands,
                    },
                    indent=2,
                )
                + "\n"
            )
        session = DeviceSession(arms, hands, config["limits"], execute=args.execute, qualification=qualification)
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
            server.bind(str(args.socket))
            os.chmod(args.socket, 0o600)
            server.listen(1)
            server.settimeout(0.005)
            deadline = time.monotonic() + 30
            while not stopping and time.monotonic() < deadline:
                arms.spin()
                arms.publish_hands(hands)
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                with conn:
                    _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        raise PermissionError("IPC peer is not the deployment owner")
                    serve_connection(
                        conn, session, stopping=lambda: stopping, pump=lambda: (arms.spin(), arms.publish_hands(hands))
                    )
                break  # One connection/plan only; no reconnect or automatic re-enable.
    finally:
        if session is not None:
            session.close()
            report = {
                "state": session.state,
                "fault": session.fault,
                "physical_stop_confirmed": session.stop_confirmed,
                "stop_errors": session.stop_errors,
                "hands_still_owned": [side for side, hand in hands.items() if hand.owns_enable],
            }
            (args.runtime.parent / "device-exit.json").write_text(json.dumps(report, indent=2) + "\n")
        else:
            for hand in hands.values():
                hand.close_readonly()
        if arms is not None:
            arms.close()
        rclpy.shutdown()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
