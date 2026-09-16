"""Offline replay, live shadow inference, and explicit finite real-machine trials."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import signal
import tempfile
import time
import uuid
import numpy as np
from .config import ROOT, load_config
from .contract import to_dp
from .control import Trajectory, settled
from .preprocess import prepare
from .wire import request
from .robot.core import ARM, Frame


def model_metadata(c):
    from .checkpoint import digest

    m = request(c["policy_url"] + "/metadata", timeout=5)
    expected = {
        "schema": "franka_wuji_dp_v1",
        "sha256": digest(c["checkpoint"]),
        "warmed_up": True,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "action_dim": 54,
        "source_hz": 30,
        "units": "radian",
        "action_semantics": "absolute_joint_target",
        "order": ["left_arm_7", "right_arm_7", "left_hand_20", "right_hand_20"],
    }
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError("Wrong/unwarmed model server or checkpoint")
    return m


def fetch(c, after=None):
    raw = request(
        c["observation_url"] + "/observation", timeout=c["observation_timeout_s"]
    )
    times = np.asarray(raw["timestamps"])
    if (
        times.shape != (2,)
        or not np.isfinite(times).all()
        or abs(times[1] - times[0] - 1 / 30) > 1e-5
    ):
        raise ValueError("Invalid observation history spacing")
    if not 0 <= time.time() - times[-1] <= c["max_observation_age_s"]:
        raise ValueError("Stale/future observation response")
    if after is not None and times[-1] <= after:
        raise ValueError("Observation did not advance after previous chunk")
    return raw


def infer(c, raw):
    obs = prepare(raw, c)
    began = time.monotonic()
    result = request(c["policy_url"] + "/infer", obs, timeout=c["inference_timeout_s"])
    if (
        "action" not in result
        or result["action"].shape != (8, 54)
        or not np.isfinite(result["action"]).all()
    ):
        raise ValueError("Invalid policy output")
    return raw, obs, result, time.monotonic() - began


def infer_live(c, after=None):
    return infer(c, fetch(c, after))


def save_result(output, index, item, trajectory=None):
    raw, obs, result, latency = item
    folder = output / f"chunk-{index:04d}"
    folder.mkdir()
    np.savez_compressed(folder / "observation.npz", **raw)
    np.savez_compressed(folder / "prediction.npz", **result)
    report = {
        "round_trip_s": latency,
        "observation_time": float(raw["timestamps"][-1]),
        "synthetic_observations": False,
    }
    if trajectory is not None:
        report.update(
            playback_seconds=trajectory.duration,
            hand_projected_values=trajectory.hand_projection_count,
        )
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def check_result_age(c, item):
    stamp = float(item[0]["timestamps"][-1])
    if (
        not 0
        <= time.time() - stamp
        <= c["max_observation_age_s"] + c["inference_timeout_s"]
    ):
        raise ValueError("Inference result is too old")


def execute(c, devices, children, output, rounds, first, metadata):
    from types import SimpleNamespace

    feedback = devices.feedback(time.monotonic())
    check_result_age(c, first)
    trajectory = Trajectory(
        first[2]["action"],
        feedback.positions,
        c,
        json.loads((output / "runtime.json").read_text())["limits"],
    )
    observed = first[0]["joint_pos"][-1]
    if np.max(np.abs(to_dp(feedback.positions)[:14] - observed[:14])) > 0.05:
        raise ValueError("Arms moved during inference")
    identity = uuid.uuid4().hex
    # The inherited final consumer checks this session digest and every frame.
    player = SimpleNamespace(
        run_id=identity,
        digest=trajectory.digest,
        plan=SimpleNamespace(start=feedback.positions),
    )
    devices.prepare(player, feedback, time.monotonic())
    try:
        check_result_age(c, first)
        current = devices.feedback(time.monotonic())
        if np.max(np.abs(current.positions[ARM] - trajectory.knots[0, ARM])) > 0.005:
            raise ValueError("Acquisition changed the planned starting pose")
    except BaseException:
        devices.stop()
        raise
    bounds = json.loads((output / "runtime.json").read_text())["limits"]
    records = []
    report = {
        "hardware_output": True,
        "completed_chunks": 0,
        "physical_stop_qualified": False,
    }
    started = time.monotonic()
    phase_started = started
    last_tick = started
    next_tick = started
    sequence = 0
    target = feedback.positions.copy()
    phase = "play"
    settled_since = None
    item = first
    future = None
    pending_save = [(0, first, trajectory)]
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            now = time.monotonic()
            if now - started > c["max_run_s"]:
                raise TimeoutError("Finite run deadline reached")
            if sequence and now - last_tick > 0.03:
                raise TimeoutError("100 Hz control loop stalled; stopping")
            children.check()
            if phase == "play":
                target, velocity = trajectory.sample(
                    0 if sequence == 0 else now - phase_started
                )
                if now - phase_started >= trajectory.duration:
                    phase = "settle"
                    phase_started = now
            else:
                velocity = np.zeros(54)
            if phase == "wait":
                if now - phase_started > c["max_hold_s"]:
                    raise TimeoutError("Inference hold deadline reached")
                if future.done():
                    item = future.result()
                    check_result_age(c, item)
                    if (
                        np.max(
                            np.abs(
                                to_dp(feedback.positions)[:14]
                                - item[0]["joint_pos"][-1, :14]
                            )
                        )
                        > 0.05
                    ):
                        raise ValueError("Arms moved during inference")
                    trajectory = Trajectory(item[2]["action"], target, c, bounds)
                    pending_save.append((report["completed_chunks"], item, trajectory))
                    phase = "play"
                    phase_started = now
                    settled_since = None
            # Hash is stable throughout this finite session; sequence never resets.
            frame = Frame(
                identity,
                player.digest,
                sequence,
                now,
                now + 0.02,
                now - started,
                target,
                velocity,
                "settle_end" if phase in ("settle", "wait") else "playback",
            )
            feedback = devices.submit(frame, now)
            if feedback is None:
                feedback = devices.feedback(time.monotonic())
            records.append(
                (now, sequence, phase, target.copy(), feedback.positions.copy())
            )
            sequence += 1
            last_tick = now
            if phase == "settle":
                if settled(feedback, target):
                    if settled_since is None:
                        settled_since = now
                    if now - settled_since >= 0.2:
                        report["completed_chunks"] += 1
                        if report["completed_chunks"] >= rounds:
                            report["finish"] = devices.finish()
                            report["state"] = "completed"
                            break
                        future = pool.submit(
                            infer_live, c, float(item[0]["timestamps"][-1])
                        )
                        phase = "wait"
                        phase_started = now
                else:
                    settled_since = None
                if now - phase_started > 5:
                    raise TimeoutError("Measured endpoint did not settle")
            next_tick += 1 / c["control_hz"]
            # No catch-up bursts after scheduler delay.
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + 1 / c["control_hz"]
            time.sleep(max(0, next_tick - time.monotonic()))
    except BaseException as e:
        report.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        # Stop before waiting on network requests or compressing logs.
        try:
            report["stop"] = devices.stop()
        except Exception as e:
            report["stop_error"] = str(e)
        pool.shutdown(wait=True, cancel_futures=True)
        for index, result, plan in pending_save:
            save_result(output, index, result, plan)
        if records:
            np.savez_compressed(
                output / "control.npz",
                monotonic=[x[0] for x in records],
                sequence=[x[1] for x in records],
                phase=[x[2] for x in records],
                command=[x[3] for x in records],
                measured=[x[4] for x in records],
            )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--shadow", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--replay", type=Path)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--supervised-trial", action="store_true")
    p.add_argument(
        "--start-bridge",
        action="store_true",
        help="Shadow only: own Wuji SDK for readback; existing arm state publishers required",
    )
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if not 1 <= a.rounds <= 1000:
        p.error("--rounds must be 1..1000")
    if a.execute and not a.supervised_trial:
        p.error("Real output requires --execute --supervised-trial")
    if a.supervised_trial and not a.execute:
        p.error("--supervised-trial requires --execute")
    if a.start_bridge and not a.shadow:
        p.error("--start-bridge applies only to --shadow")
    c = load_config()
    output = (
        a.output
        or ROOT
        / "logs"
        / ("dp-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "deploy-config.json").write_text(json.dumps(c, indent=2) + "\n")
    from .hardware import runtime, commands, Children, preflight_owners

    if not (a.execute or a.shadow or a.replay):
        runtime(c, output, False)
        preview = commands(c, output, Path("/tmp/dp-runtime-preview"), True)
        (output / "commands-preview.json").write_text(
            json.dumps(preview, indent=2) + "\n"
        )
        print("Check only, no model/device connections:", output)
        return
    metadata = model_metadata(c)
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if a.replay:
        with np.load(a.replay, allow_pickle=False) as data:
            raw = {k: data[k] for k in data.files}
        item = infer(c, raw)
        save_result(output, 0, item)
        (output / "report.json").write_text(
            json.dumps(
                {
                    "state": "replayed",
                    "hardware_output": False,
                    "output_shape": list(item[2]["action"].shape),
                },
                indent=2,
            )
            + "\n"
        )
        print("Replay saved:", output)
        return

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    # Fail before launching any device if the local collector port is occupied.
    import socket
    from urllib.parse import urlparse

    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", urlparse(c["observation_url"]).port))
    with ExitStack() as stack:
        ROOT.joinpath(".deployment").mkdir(exist_ok=True)
        lock = stack.enter_context((ROOT / ".deployment/owner.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.execute or a.start_bridge:
            preflight_owners(a.execute)
        runtime(c, output, a.execute)
        ipc = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="dp-devices-"))
        )
        children = Children(output)
        stack.callback(children.close)
        cmds = commands(c, output, ipc, a.execute, bridge=a.execute or a.start_bridge)
        (output / "commands.json").write_text(json.dumps(cmds, indent=2) + "\n")
        for label, cmd in cmds.items():
            children.launch(label, cmd)
        devices = None
        deadline = time.monotonic() + 45
        if a.execute or a.start_bridge:
            while not (ipc / "devices.sock").exists():
                children.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("Device bridge did not start")
                time.sleep(0.1)
            from .robot.ipc import RemoteDevices

            devices = RemoteDevices(
                ipc / "devices.sock",
                execute=a.execute,
                finish_policy="disable",
                arm_speed_rad_s=c["arm_speed_rad_s"],
            )
            stack.callback(devices.close)
            if a.execute:
                stack.callback(devices.stop)
            inventory = devices.call("inventory", timeout=3)
            (output / "devices.json").write_text(json.dumps(inventory, indent=2) + "\n")
        last_error = None
        while time.monotonic() < deadline:
            children.check()
            try:
                raw = fetch(c)
                break
            except Exception as e:
                last_error = str(e)
                time.sleep(0.1)
        else:
            raise TimeoutError(
                "No complete live RGBD/state history: " + str(last_error)
            )
        first = infer(c, raw)
        if a.execute:
            execute(c, devices, children, output, a.rounds, first, metadata)
        else:
            from .hardware import limits

            bounds, _ = limits(c)
            from .contract import to_device

            reports = []
            for i in range(a.rounds):
                item = (
                    first if i == 0 else infer_live(c, float(item[0]["timestamps"][-1]))
                )
                plan = None
                error = None
                try:
                    plan = Trajectory(
                        item[2]["action"],
                        to_device(item[0]["joint_pos"][-1]),
                        c,
                        bounds,
                    )
                except ValueError as e:
                    error = str(e)
                save_result(output, i, item, plan)
                reports.append(
                    {"chunk": i, "admissible": error is None, "reason": error}
                )
            (output / "report.json").write_text(
                json.dumps(
                    {"state": "shadow", "hardware_output": False, "chunks": reports},
                    indent=2,
                )
                + "\n"
            )
        print("Saved:", output)


if __name__ == "__main__":
    main()
