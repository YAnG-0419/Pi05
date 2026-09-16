"""Contract, stale-data and framing tests; these never connect to hardware."""

import io
from pathlib import Path
import socket
import threading

import numpy as np
from observation import IMAGE_KEYS
from observation import IMAGE_SHAPES
from observation import STATE_ORDER
from observation import ObservationBuffer
from observation import ObservationUnavailableError
from observation import joint_names
from observation import positions_in_order
from observation import read_record
from observation import send_record
import pytest
import yaml


def populate(buffer, *, wall=100, mono=10, stamps=None):
    stamps = stamps or {}
    for index, source in enumerate(STATE_ORDER):
        names = joint_names(source)
        values = np.arange(len(names)) + index * 100
        record = {
            "source": source,
            "names": names[::-1],
            "positions": values[::-1],
            "stamp": stamps.get(source, wall - 0.1),
            "units": "radian",
            "measured": True,
            "stamp_basis": "test",
        }
        buffer.add(record, now_wall=wall, now_mono=mono)
    for i, source in enumerate(IMAGE_KEYS):
        # Known RGB swatch: ensure the adapter never reverses R/B or rescales.
        image = np.zeros(IMAGE_SHAPES[source], np.uint8)
        image[:] = [200, 10, i]
        buffer.add(
            {
                "source": source,
                "shape": list(image.shape),
                "encoding": "rgb8",
                "stamp": stamps.get(source, wall - 0.1),
                "stamp_basis": "test",
            },
            image.tobytes(),
            now_wall=wall,
            now_mono=mono,
        )


def test_contract_matches_training_108_state_and_rgb():
    from openpi.models.model import ModelType
    from openpi.policies.fr3_wuji_policy import Fr3WujiInputs

    buffer = ObservationBuffer()
    populate(buffer)
    obs, metadata = buffer.snapshot(now_wall=100, now_mono=10)
    state108 = np.zeros(108, dtype=np.float32)
    state108[0:7] = np.arange(7)
    state108[28:48] = np.arange(20) + 100
    state108[14:21] = np.arange(7) + 200
    state108[68:88] = np.arange(20) + 300
    transform = Fr3WujiInputs(model_type=ModelType.PI05)
    actual = transform(obs)
    expected = transform({**obs, "observation/state": state108})
    np.testing.assert_array_equal(actual["state"], expected["state"])
    assert obs["observation/state"].dtype == np.float32
    assert obs["observation/state"].shape == (54,)
    for i, key in enumerate(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")):
        np.testing.assert_array_equal(actual["image"][key][0, 0], [200, 10, i])
    assert metadata["skew_seconds"] == 0


def test_names_are_not_guessed_or_silently_padded():
    with pytest.raises(ValueError, match="joint names"):
        positions_in_order("left_arm", joint_names("right_arm"), np.zeros(7))
    with pytest.raises(ValueError, match="joint names"):
        positions_in_order("left_hand", ["l_thumb_ip"] * 20, np.zeros(20))
    with pytest.raises(ValueError, match="finite measured positions"):
        positions_in_order("left_arm", joint_names("left_arm"), [float("nan")] * 7)


def test_names_match_existing_workcell_contract():
    path = Path("/home/user/lpy/gello-retarget/data_collection/contract/dataset_contract.yaml")
    if not path.exists():
        pytest.skip("This workcell's read-only reference is not present")
    expected = yaml.safe_load(path.read_text())["contract_joint_names"]
    for source in STATE_ORDER:
        assert joint_names(source) == tuple(expected[source])


def test_missing_stale_skew_clock_jump_and_duplicate_rejected():
    buffer = ObservationBuffer()
    with pytest.raises(ObservationUnavailableError, match="missing"):
        buffer.snapshot(now_wall=100, now_mono=10)
    populate(buffer)
    with pytest.raises(ObservationUnavailableError, match="stale"):
        buffer.snapshot(now_wall=100.3, now_mono=10.3)
    with pytest.raises(ObservationUnavailableError, match="clock_jump"):
        buffer.snapshot(now_wall=102, now_mono=10)
    other = ObservationBuffer()
    populate(other, stamps={"cam2": 99.99})
    with pytest.raises(ObservationUnavailableError, match="timestamp_skew"):
        other.snapshot(now_wall=100, now_mono=10)
    with pytest.raises(ValueError, match="timestamp"):
        buffer.add({"source": "left_arm", "stamp": 99.9}, now_wall=100, now_mono=10)
    with pytest.raises(ObservationUnavailableError, match="missing:left_arm"):
        buffer.snapshot(now_wall=100, now_mono=10)


def test_delayed_packet_is_not_made_fresh_by_receipt():
    buffer = ObservationBuffer()
    populate(buffer)
    with pytest.raises(ValueError, match="old source timestamp"):
        buffer.add({"source": "cam0", "stamp": 98}, now_wall=100, now_mono=10)
    with pytest.raises(ObservationUnavailableError, match="missing:cam0"):
        buffer.snapshot(now_wall=100, now_mono=10)


def test_selects_historical_state_near_image_not_latest_state():
    buffer = ObservationBuffer()
    populate(buffer)
    for source in STATE_ORDER:
        buffer.add(
            {
                "source": source,
                "stamp": 100,
                "stamp_basis": "test",
                "measured": True,
                "units": "radian",
                "names": joint_names(source),
                "positions": [999] * len(joint_names(source)),
            },
            now_wall=100,
            now_mono=10,
        )
    obs, _ = buffer.snapshot(now_wall=100, now_mono=10)
    assert not np.any(obs["observation/state"] == 999)


def test_binary_rgb_transport_and_truncated_frame():
    left, right = socket.socketpair()
    left.settimeout(3)
    right.settimeout(3)
    payload = bytes(range(256)) * 3600
    with left, right:
        thread = threading.Thread(target=send_record, args=(left, {"source": "cam1"}, payload))
        thread.start()
        with right.makefile("rb") as stream:
            record, received = read_record(stream)
        thread.join()
    assert received == payload
    assert record["payload_bytes"] == len(payload)
    with pytest.raises(EOFError, match="disconnected"):
        read_record(io.BytesIO(b"\x00\x00"))


def test_latency_diagnostics_do_not_replace_source_timestamps():
    buffer = ObservationBuffer()
    populate(buffer)
    _, metadata = buffer.snapshot(now_wall=100.02, now_mono=10.02)
    for source in (*STATE_ORDER, *IMAGE_KEYS):
        sample = metadata["samples"][source]
        assert sample["stamp"] == pytest.approx(99.9)
        assert sample["age_seconds"] == pytest.approx(0.12)
        assert sample["age_at_buffer_receipt_seconds"] == pytest.approx(0.1)
        assert sample["buffer_wait_seconds"] == pytest.approx(0.02)
    with pytest.raises(ObservationUnavailableError):
        buffer.snapshot(now_wall=100.3, now_mono=10.3)


def test_rgb_profile_preserves_rgb_contract_and_reference_files(tmp_path):
    import observe

    camera_path = observe.REFERENCE / "data_collection/config/cameras.yaml"
    cameras = yaml.safe_load(camera_path.read_text())["camera_bringup"]
    original = camera_path.read_bytes()
    path, changes = observe.prepare_camera_profile(cameras, tmp_path, "rgb")
    assert path == "/logs/camera-config/cameras.yaml"
    assert camera_path.read_bytes() == original
    deployed = yaml.safe_load((tmp_path / "camera-config/cameras.yaml").read_text())["camera_bringup"]
    for key in IMAGE_KEYS:
        before = yaml.safe_load(Path(changes[key]["source"]).read_text())
        after = yaml.safe_load((tmp_path / f"camera-config/{key}.yaml").read_text())
        for name, value in before.items():
            if name not in changes[key]["overrides"]:
                assert after[name] == value
        assert after["enable_depth"] is False
        assert after["enable_frame_sync"] is False
        assert after["frame_aggregate_mode"] == "disable"
        assert deployed[key]["serial_number"] == cameras[key]["serial_number"]
    recording, changes = observe.prepare_camera_profile(cameras, tmp_path, "recording")
    assert recording.endswith("data_collection/config/cameras.yaml")
    assert not changes
