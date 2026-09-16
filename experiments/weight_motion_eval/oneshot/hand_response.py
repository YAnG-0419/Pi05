"""Read-only capture or one supervised small joint excursion; no model/arms."""

import argparse
import copy
import json
import math
from pathlib import Path
import signal
import socket
import time
import uuid

import numpy as np

from .hand import HandOwner
from .hand import stabilize_hands
from .limits import HAND_SPEED_RAD_S
from .limits import SLIDER_POSITION_TOLERANCE_RAD
from .limits import SLIDER_SPEED_RAD_S
from .qualification import SupervisedTrial


class JointTrial(SupervisedTrial):
    def __init__(self, side, identity):
        super().__init__()
        self.side, self.identity_snapshot = side, copy.deepcopy(identity)

    def check_hand(self, side, identity):
        if side != self.side or identity != self.identity_snapshot:
            raise ValueError("Single-joint test identity/parameters changed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--address")
    parser.add_argument("--execute", action="store_true", help="Enable selected hand and move one joint")
    parser.add_argument("--hand-control", choices=("slider", "strict"), default="slider")
    parser.add_argument(
        "--keep-current-parameters", action="store_true", help="Skip deployment parameter setup for a readback baseline"
    )
    parser.add_argument("--joint", type=int, default=0, help="Canonical hand index 0..19, not firmware nid")
    parser.add_argument("--delta-deg", type=float, default=2.0)
    parser.add_argument("--speed-deg-s", type=float)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--effort-a", type=float)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/weight_motion_eval") / ("hand-response-" + uuid.uuid4().hex[:10])
    )
    args = parser.parse_args(argv)
    if not 0 <= args.joint < 20:
        parser.error("--joint must be 0..19")
    if not math.isfinite(args.delta_deg) or not 0 < abs(args.delta_deg) <= 3:
        parser.error("Excursion must be nonzero and at most 3 degrees")
    cap = math.degrees(SLIDER_SPEED_RAD_S) if args.hand_control == "slider" else 10.0
    args.speed_deg_s = (
        args.speed_deg_s if args.speed_deg_s is not None else (cap if args.hand_control == "slider" else 5.0)
    )
    if not math.isfinite(args.speed_deg_s) or not 0 < args.speed_deg_s <= cap:
        parser.error(f"Test target speed must be at most {cap:g} degrees/s")
    for key, ceiling in (("kp", 8.0), ("kd", 0.1), ("effort_a", 1.0)):
        value = getattr(args, key)
        if value is not None and (not args.execute or not math.isfinite(value) or not 0 < value <= ceiling):
            parser.error(f"--{key.replace('_', '-')} requires --execute and 0 < value <= {ceiling}")
    args.address = args.address or {"left": "192.168.1.110:7447", "right": "192.168.2.111:7447"}[args.side]
    return args


def excursion(elapsed, delta, duration):
    """Rest-to-rest quintic, 0.5 s hold at excursion and initial position."""
    if elapsed < duration:
        u = max(0.0, elapsed / duration)
        return delta * (10 * u**3 - 15 * u**4 + 6 * u**5)
    if elapsed < duration + 0.5:
        return delta
    if elapsed < 2 * duration + 0.5:
        u = (elapsed - duration - 0.5) / duration
        return delta * (1 - (10 * u**3 - 15 * u**4 + 6 * u**5))
    return 0.0


def stationary(owner, seconds=0.5, timeout=3):
    end, since = time.monotonic() + timeout, None
    while time.monotonic() < end:
        try:
            q, dq, _ = owner.poll(time.monotonic(), time.time())
        except ValueError as error:
            if str(error) not in {"No fresh measured hand state", "No fresh hand diagnostics"}:
                raise
            time.sleep(0.005)
            continue
        if getattr(owner, "hand_control", "strict") == "slider":
            return q
        if np.max(np.abs(dq)) <= 0.02:
            since = since or time.monotonic()
            if time.monotonic() - since >= seconds:
                return q
        else:
            since = None
        time.sleep(0.005)
    raise RuntimeError("Hand did not confirm stationary feedback")


def apply_parameters(owner, args, baseline):
    """Disabled-only changes; untouched joints retain their readback values."""
    if not any(getattr(args, key) is not None for key in ("kp", "kd", "effort_a")):
        return
    if any(j.status_word.ext_state != 1 for j in owner.diagnostics[0].values()):
        raise RuntimeError("Parameter test requires all joints disabled and fault-free")
    gains = [(g["kp"], g["kd"]) for g in baseline["mit_gains"]]
    kp, kd = gains[args.joint]
    if args.effort_a is not None and args.effort_a > baseline["effort_limits"][args.joint]:
        raise ValueError("Initial diagnostic test cannot raise current cap above current readback")
    gains[args.joint] = (args.kp if args.kp is not None else kp, args.kd if args.kd is not None else kd)
    owner.trace.add("parameters_write_attempt", joint=args.joint, kp=args.kp, kd=args.kd, effort_a=args.effort_a)
    if args.kp is not None or args.kd is not None:
        owner.hand.mit_params().set(gains)
    if args.effort_a is not None:
        owner.hand.joint(args.joint).effort_limit().set(args.effort_a)
    readback = owner.identity()
    for actual, wanted in zip(readback["mit_gains"], gains, strict=True):
        if not np.allclose([actual["kp"], actual["kd"]], wanted, rtol=1e-6, atol=1e-7):
            raise RuntimeError("MIT parameter readback mismatch")
    expected = baseline["effort_limits"].copy()
    if args.effort_a is not None:
        expected[args.joint] = args.effort_a
    if not np.allclose(readback["effort_limits"], expected, rtol=1e-6, atol=1e-7):
        raise RuntimeError("Current limit readback mismatch")
    owner.trace.add("parameters_readback", identity=readback)


def main(argv=None):
    args = parse_args(argv)
    # Probe permissions before constructing a Zenoh runtime. No packets sent.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM):
            pass
    except PermissionError as error:
        raise SystemExit("Network access blocked; no hand queried or enabled: " + str(error)) from error
    import wuji_sdk as sdk

    args.output.mkdir(parents=True, exist_ok=False)
    owner = None
    baseline = None
    changed = False
    cancelled = False
    report = {
        "mode": "execute" if args.execute else "read_only",
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "physical_stop_confirmed": False,
    }

    def interrupt(*_):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        owner = HandOwner(sdk, args.side, args.address, hand_control=args.hand_control)
        baseline = owner.identity()
        report["baseline"] = baseline
        if not args.execute:
            end = time.monotonic() + 3
            while time.monotonic() < end and not cancelled:
                owner.capture_raw()
                time.sleep(0.005)
            report["state"] = "captured"
            return
        start = stationary(owner)
        if any(j.status_word.ext_state != 1 for j in owner.diagnostics[0].values()):
            raise RuntimeError("All selected hand joints must be Ready; no automatic fault clear")
        from experiments.weight_motion_eval.reference import hand_position_limits

        limits = hand_position_limits(args.side)
        lower, upper = limits["lower"], limits["upper"]
        report["reference_hashes"] = limits["sources"]
        delta = math.radians(args.delta_deg)
        target = start.copy()
        target[args.joint] += delta
        if np.any(np.minimum(start, target) < lower) or np.any(np.maximum(start, target) > upper):
            raise ValueError("Single-joint test exceeds reference position limits")
        report["joint_name"] = limits["names"][args.joint]
        if cancelled:
            raise RuntimeError("Operator cancelled preparation")
        if not args.keep_current_parameters:
            report["deployment_parameters"] = owner.configure_deployment()
        # Temporary per-joint overrides restore to this configured deployment
        # baseline. The entry readback remains in report['baseline'].
        baseline = owner.identity()
        changed = any(getattr(args, key) is not None for key in ("kp", "kd", "effort_a"))
        if cancelled:
            raise RuntimeError("Operator cancelled preparation")
        apply_parameters(owner, args, baseline)
        report["operating_parameters"] = owner.identity()
        # Reacquire pose after potentially slow parameter RPCs.
        current = stationary(owner)
        if args.hand_control == "strict" and np.max(np.abs(current - start)) > 0.005:
            raise RuntimeError("Pose changed during preparation")
        owner.enable(qualification=JointTrial(args.side, owner.identity()), cancelled=lambda: cancelled)
        if args.hand_control == "strict":
            stabilize_hands([owner], cancelled=lambda: cancelled)
        current, velocity, _ = owner.poll(time.monotonic(), time.time())
        if args.hand_control == "strict" and (
            np.max(np.abs(current - start)) > 0.005 or np.max(np.abs(velocity)) > 0.02
        ):
            raise RuntimeError("Enable changed pose or produced motion")
        duration = max(1.0, 1.875 * abs(delta) / math.radians(args.speed_deg_s))
        if args.hand_control == "slider":
            start = owner.last[0].copy()
            target = start.copy()
            target[args.joint] += delta
            if np.any(np.minimum(start, target) < lower) or np.any(np.maximum(start, target) > upper):
                raise ValueError("Updated slider start/target exceeds model limits")
            duration = max(0.02, abs(delta) / math.radians(args.speed_deg_s))
        begun = time.monotonic()
        owner.last = (current.copy(), begun - 0.01)
        while time.monotonic() - begun <= 2 * duration + 1.0:
            if cancelled:
                raise RuntimeError("Operator cancelled test")
            now = time.monotonic()
            q = start.copy()
            q[args.joint] += excursion(now - begun, delta, duration)
            if args.hand_control == "slider":
                elapsed = now - begun
                progress = (
                    min(1.0, elapsed / duration)
                    if elapsed < duration + 0.5
                    else max(0.0, 1 - (elapsed - duration - 0.5) / duration)
                )
                q[args.joint] = start[args.joint] + delta * progress
            measured, velocity, _ = owner.poll(now, time.time())
            if args.hand_control == "strict" and np.max(np.abs(velocity)) > HAND_SPEED_RAD_S:
                raise RuntimeError("Measured hand velocity exceeds 45 degrees/s")
            owner.submit(q, created=now, valid_until=now + 0.02, now=now, lower=lower, upper=upper)
            time.sleep(max(0.0, 0.01 - (time.monotonic() - now)))
        # Stream the return target until measured settling (max three seconds).
        end, settled = time.monotonic() + 3, None
        while time.monotonic() < end:
            if cancelled:
                raise RuntimeError("Operator cancelled test")
            now = time.monotonic()
            measured, velocity, _ = owner.poll(now, time.time())
            owner.submit(start, created=now, valid_until=now + 0.02, now=now, lower=lower, upper=upper)
            reached = np.max(np.abs(measured - start)) <= 0.01 and np.max(np.abs(velocity)) <= 0.02
            if args.hand_control == "slider":
                reached = (
                    np.max(np.abs(measured - start)) <= SLIDER_POSITION_TOLERANCE_RAD
                    and np.max(np.abs(owner.last[0] - start)) <= 1e-6
                )
            if reached:
                settled = settled or now
                if now - settled >= 0.5:
                    owner.disable_after_stop(time.monotonic())
                    _, final_velocity, _ = owner.poll(time.monotonic(), time.time())
                    report.update(
                        state="completed", physical_stop_confirmed=bool(np.max(np.abs(final_velocity)) <= 0.02)
                    )
                    break
            else:
                settled = None
            time.sleep(max(0.0, 0.01 - (time.monotonic() - now)))
        else:
            raise RuntimeError("Return pose did not settle")
    except BaseException as error:
        report.update(state="failed", error=str(error))
        if owner is not None:
            owner.trace.trigger(error)
        raise
    finally:
        if owner is not None:
            try:
                if owner.owns_enable:
                    owner.emergency_stop()
                    end = time.monotonic() + 2
                    while time.monotonic() < end:
                        try:
                            owner.capture_raw()
                        except Exception as error:
                            owner.trace.add("capture_error", error=str(error))
                        time.sleep(0.005)
                    try:
                        stationary(owner)
                        owner.disable_after_stop(time.monotonic())
                        _, final_velocity, _ = owner.poll(time.monotonic(), time.time())
                        report["physical_stop_confirmed"] = bool(np.max(np.abs(final_velocity)) <= 0.02)
                    except Exception as error:
                        report["release_error"] = str(error)
                if changed:
                    if owner.owns_enable:
                        report["parameters_restored"] = False
                    else:
                        # Never restore while enabled or in a firmware-fault state.
                        stationary(owner)
                        if any(j.status_word.ext_state != 1 for j in owner.diagnostics[0].values()):
                            raise RuntimeError("Cannot restore parameters outside Ready state")
                        owner.hand.mit_params().set([(g["kp"], g["kd"]) for g in baseline["mit_gains"]])
                        owner.hand.joint(args.joint).effort_limit().set(baseline["effort_limits"][args.joint])
                        restored = owner.identity()
                        report["parameters_restored"] = restored == baseline
                        owner.trace.add("parameters_restored", identity=restored)
                        if not report["parameters_restored"]:
                            raise RuntimeError("Restoration readback mismatch")
            except Exception as error:
                report["cleanup_error"] = str(error)
                report["parameters_restored"] = False if changed else None
            finally:
                report["hand_still_owned"] = owner.owns_enable
                try:
                    owner.trace.write(args.output / ("hand-raw-" + args.side + ".jsonl"), report.get("baseline"))
                finally:
                    try:
                        if not owner.owns_enable:
                            owner.close_readonly()
                    finally:
                        (args.output / "response-report.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(report, indent=2))
                if report.get("state") == "completed" and report.get("cleanup_error"):
                    raise RuntimeError("Motion completed but parameter cleanup failed; inspect response-report.json")
        else:
            (args.output / "response-report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
