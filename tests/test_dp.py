import json
from pathlib import Path
from types import SimpleNamespace as S
import time
import numpy as np
import pytest
from dp_deploy.config import load_config
from dp_deploy.contract import to_device, to_dp, ordered_positions, joint_names
from dp_deploy.control import Trajectory
from dp_deploy.observe import Buffer, ObservationUnavailable, decode_image
from dp_deploy.preprocess import prepare, resize
from dp_deploy.robot.core import ARM, ConsumerGuard, Feedback, Frame


def test_joint_permutation_has_independent_explicit_expected_values():
    dp = np.r_[np.full(7, 1), np.full(7, 2), np.full(20, 3), np.full(20, 4)]
    np.testing.assert_array_equal(
        to_device(dp),
        np.r_[np.full(7, 1), np.full(20, 3), np.full(7, 2), np.full(20, 4)],
    )
    np.testing.assert_array_equal(to_dp(to_device(dp)), dp)


def test_named_positions_reorder_and_reject_missing():
    names = joint_names("right_hand")
    np.testing.assert_array_equal(
        ordered_positions("right_hand", names[::-1], np.arange(20)[::-1]), np.arange(20)
    )
    with pytest.raises(ValueError):
        ordered_positions("right_hand", names[:-1], np.arange(19))


def raw():
    return {
        "joint_pos": np.zeros((2, 54), np.float32),
        "head_rgb": np.full((2, 4, 6, 3), 128, np.uint8),
        "left_wrist_rgb": np.full((2, 4, 6, 3), 64, np.uint8),
        "right_wrist_rgb": np.full((2, 4, 6, 3), 255, np.uint8),
        "head_depth": np.full((2, 4, 6), 1000, np.uint16),
    }


def test_depth_mm_and_metres_identical_with_training_scaling():
    r = raw()
    a = prepare(r, load_config())
    r["head_depth"] = np.ones((2, 4, 6), np.float32)
    b = prepare(r, load_config())
    np.testing.assert_array_equal(a["head_depth"], b["head_depth"])
    assert a["head_depth"].max() == 0.5
    r["head_depth"][:] = np.nan
    with pytest.raises(ValueError, match="no valid"):
        prepare(r, load_config())


def test_rgb_and_depth_letterbox_match_training_reference_without_importing_conversion_stack():
    import ast

    path = Path("/home/descfly/lcl_work/Convert_data/src/convest/dp/convert.py")
    if not path.exists():
        pytest.skip("Training source absent on this machine")
    tree = ast.parse(path.read_text())
    func = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resize"
    )
    import av, math

    ns = {"np": np, "av": av, "math": math}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(path), "exec"), ns)
    rng = np.random.default_rng(1)
    for h, w in [(400, 640), (480, 640), (17, 11)]:
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        depth = rng.random((h, w), dtype=np.float32)
        np.testing.assert_array_equal(resize(rgb), ns["resize"](rgb, 320, 240))
        np.testing.assert_array_equal(
            resize(depth, depth=True), ns["resize"](depth, 320, 240, True)
        )


def test_ros_bgr_padding_and_depth_endianness():
    m = S(
        encoding="bgr8",
        height=1,
        width=2,
        step=8,
        is_bigendian=False,
        data=bytes([1, 2, 3, 4, 5, 6, 0, 0]),
    )
    np.testing.assert_array_equal(decode_image(m), [[[3, 2, 1], [6, 5, 4]]])
    m = S(
        encoding="16UC1",
        height=1,
        width=2,
        step=6,
        is_bigendian=True,
        data=np.array([1000, 2000, 0], dtype=">u2").tobytes(),
    )
    np.testing.assert_array_equal(decode_image(m, True), [[1000, 2000]])


def populated():
    b = Buffer(load_config())
    b.offset = 900
    for key in b.buffers:
        n = 7 if key.endswith("arm") else 20 if key.endswith("hand") else 1
        for t in [99.9, 99.94, 99.98, 100.01]:
            b.add(key, 900 + t, np.full(n, t, np.float32), received=t)
    return b


def test_observations_are_causal_30hz_history_and_reject_stale():
    b = populated()
    r = b.snapshot(wall=1000.02, mono=100.02)
    assert r["joint_pos"].shape == (2, 54)
    np.testing.assert_allclose(np.diff(r["timestamps"]), 1 / 30)
    assert np.all(r["source_timestamps"] <= r["timestamps"][None] + 1e-7)
    with pytest.raises(ObservationUnavailable):
        b.snapshot(wall=1001, mono=101)
    with pytest.raises(ObservationUnavailable, match="clock"):
        b.snapshot(wall=1001, mono=100.02)


def test_missing_history_and_reversed_source_clock():
    b = Buffer(load_config())
    with pytest.raises(ObservationUnavailable):
        b.snapshot()
    b = populated()
    with pytest.raises(ValueError, match="backwards"):
        b.add("left_arm", 999, np.zeros(7))
    assert all(not q for q in b.buffers.values())


def plan_inputs():
    c = load_config()
    bounds = {"lower": [-3] * 54, "upper": [3] * 54}
    start = np.zeros(54)
    actions = np.zeros((8, 54))
    actions[:, 7] = np.linspace(0.01, 0.03, 8)
    return c, bounds, start, actions


def test_trajectory_right_arm_mapping_limits_speed_and_endpoints():
    c, bounds, start, actions = plan_inputs()
    p = Trajectory(actions, start, c, bounds)
    q, dq = p.sample(p.duration)
    assert q[27] == pytest.approx(0.03)
    assert q[7] == 0
    previous = p.sample(0)[0]
    for t in np.linspace(0, p.duration, 500):
        q, v = p.sample(t)
        assert np.max(np.abs(v[ARM])) <= c["target_arm_speed_rad_s"] + 1e-9
    np.testing.assert_array_equal(dq, np.zeros(54))
    actions[0, 0] = 1
    with pytest.raises(ValueError, match="too far"):
        Trajectory(actions, start, c, bounds)
    actions[0, 0] = 4
    with pytest.raises(ValueError, match="limits"):
        Trajectory(actions, start, c, bounds)


def test_final_boundary_rejects_expiry_stale_feedback_and_sequence():
    q = np.zeros(54)
    fb = Feedback(q, q, (10,) * 4, (10,) * 4)
    g = ConsumerGuard([-3] * 54, [3] * 54)
    g.arm("id", "digest", fb, 10)
    f = Frame("id", "digest", 0, 10, 10.02, 0, q, q, "playback")
    g.validate(f, fb, 10.001)
    g.commit(f)
    with pytest.raises(ValueError):
        g.validate(f, fb, 10.03)
    with pytest.raises(ValueError):
        g.validate(f, fb, 10.005)
    with pytest.raises(ValueError):
        fb.check(10.151, 0)


class FakeDevices:
    def __init__(self):
        self.q = np.zeros(54)
        self.stopped = False
        self.submitted = []

    def feedback(self, now):
        return Feedback(
            self.q, np.zeros(54), (now,) * 4, (now,) * 4, hand_control="slider"
        )

    def prepare(self, p, feedback, now):
        self.guard = ConsumerGuard([-3] * 54, [3] * 54, hand_control="slider")
        self.guard.arm(p.run_id, p.digest, feedback, time.monotonic())

    def submit(self, frame, now):
        fb = self.feedback(time.monotonic())
        self.guard.validate(frame, fb, time.monotonic())
        self.guard.commit(frame)
        self.q = frame.positions.copy()
        self.submitted.append(frame)
        return self.feedback(time.monotonic())

    def finish(self):
        return {"state": "released"}

    def stop(self):
        self.stopped = True
        return {"state": "stopped"}


def test_finite_executor_runs_original_consumer_guard_and_stops(tmp_path):
    from dp_deploy.client import execute

    c, bounds, start, actions = plan_inputs()
    dev = FakeDevices()
    (tmp_path / "runtime.json").write_text(json.dumps({"limits": bounds}))
    r = raw()
    r["timestamps"] = np.array([time.time() - 1 / 30, time.time()])
    r["joint_pos"] = np.zeros((2, 54), np.float32)
    item = (r, {}, {"action": actions}, 0.01)
    execute(c, dev, S(check=lambda: None), tmp_path, 1, item, {})
    assert dev.stopped and len(dev.submitted) > 30
    assert [f.sequence for f in dev.submitted] == list(range(len(dev.submitted)))
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["state"] == "completed"
    assert dev.q[27] == pytest.approx(0.03)


def test_inference_failure_stops_active_session(tmp_path, monkeypatch):
    from dp_deploy import client

    c, bounds, start, actions = plan_inputs()
    dev = FakeDevices()
    (tmp_path / "runtime.json").write_text(json.dumps({"limits": bounds}))
    r = raw()
    r["timestamps"] = np.array([time.time() - 1 / 30, time.time()])
    r["joint_pos"] = np.zeros((2, 54), np.float32)

    def fail(*args):
        raise TimeoutError("injected model timeout")

    monkeypatch.setattr(client, "infer_live", fail)
    with pytest.raises(TimeoutError, match="injected"):
        client.execute(
            c,
            dev,
            S(check=lambda: None),
            tmp_path,
            2,
            (r, {}, {"action": actions}, 0.01),
            {},
        )
    assert dev.stopped
    assert json.loads((tmp_path / "report.json").read_text())["state"] == "failed"


def test_native_controller_deadline_guard(tmp_path):
    import shutil, subprocess

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    root = Path(__file__).resolve().parents[1]
    code = tmp_path / "deadline_test.cpp"
    code.write_text("""#include "dp_deploy/robot/deadline.hpp"
#include <cassert>
int main() {
 pi05::Envelope e;
 assert(pi05::decode("pi05v1/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/0/1020000000", e));
 assert(e.sequence == 0 && e.expiry == 1020000000);
 assert(pi05::fresh(1000000000,1020000000,1010000000));
 assert(!pi05::fresh(1000000000,1020000000,1020000000));
 assert(!pi05::fresh(1000000000,1020000000,999999999));
 assert(!pi05::fresh(1000000000,1030000000,1010000000));
 assert(!pi05::decode("pi05v1/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/-1/1020000000", e));
 assert(!pi05::decode("foreign/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/0/1020000000", e));
}
""")
    binary = tmp_path / "deadline_test"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(root), str(code), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_slow_acquisition_cannot_reuse_expired_inference(tmp_path, monkeypatch):
    from dp_deploy import client

    c, bounds, start, actions = plan_inputs()
    (tmp_path / "runtime.json").write_text(json.dumps({"limits": bounds}))
    now = time.time()
    r = raw()
    r["timestamps"] = np.array([now - 1 / 30, now])

    class SlowDevices(FakeDevices):
        def prepare(self, p, feedback, stamp):
            super().prepare(p, feedback, stamp)
            monkeypatch.setattr(client.time, "time", lambda: now + 10)

    dev = SlowDevices()
    with pytest.raises(ValueError, match="too old"):
        client.execute(
            c,
            dev,
            S(check=lambda: None),
            tmp_path,
            1,
            (r, {}, {"action": actions}, 0.01),
            {},
        )
    assert dev.stopped and not dev.submitted


def test_wrong_server_contract_rejected_before_devices(monkeypatch):
    from dp_deploy import client, checkpoint

    monkeypatch.setattr(checkpoint, "digest", lambda p: "expected")
    monkeypatch.setattr(
        client,
        "request",
        lambda *args, **kwargs: {"schema": "pi05", "sha256": "expected"},
    )
    with pytest.raises(ValueError, match="Wrong"):
        client.model_metadata(load_config())
