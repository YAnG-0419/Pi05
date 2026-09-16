"""Hardware-independent observation contract and timestamp matching. No commands."""

from collections import deque
from dataclasses import dataclass
import json
import math
import struct
import threading
import time

import numpy as np

STATE_ORDER = ("left_arm", "left_hand", "right_arm", "right_hand")
IMAGE_KEYS = {
    "cam0": "observation/image",
    "cam1": "observation/left_wrist_image",
    "cam2": "observation/right_wrist_image",
}
IMAGE_SHAPES = {"cam0": (400, 640, 3), "cam1": (480, 640, 3), "cam2": (480, 640, 3)}
SOURCES = (*STATE_ORDER, *IMAGE_KEYS)
PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand "
    "and place it in the left basket."
)


def joint_names(source):
    side, part = source.split("_")
    if part == "arm":
        return tuple(f"{side}_fr3_joint{i}" for i in range(1, 8))
    prefix = "l" if side == "left" else "r"
    suffixes = ["thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip"]
    for finger in ("index", "middle", "ring", "pinky"):
        stem = finger if finger == "pinky" else finger + "_finger"
        suffixes.extend(f"{stem}_{joint}" for joint in ("mcp_flex", "mcp_abd", "pip", "dip"))
    return tuple(f"{prefix}_{name}" for name in suffixes)


def positions_in_order(source, names, values):
    names = tuple(names)
    expected = joint_names(source)
    values = np.asarray(values, dtype=np.float32)
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(f"{source}: missing, duplicate, foreign, or extra joint names")
    if values.shape != (len(expected),) or not np.isfinite(values).all():
        raise ValueError(f"{source}: expected {len(expected)} finite measured positions")
    return values[[names.index(name) for name in expected]].copy()


@dataclass(frozen=True)
class Sample:
    stamp: float
    received: float
    value: np.ndarray
    basis: str
    producer_received: float | None = None


class ObservationUnavailableError(ValueError):
    pass


class ObservationBuffer:
    """Match near the oldest latest image, using bounded histories of measured states.

    All stamps must use the host Unix clock. FR3 direct reads use host receive
    time (not the robot's unrelated boot clock); this is approximate alignment.
    """

    def __init__(self, *, max_age=0.2, max_skew=0.08, state_live_age=0.15):
        if not all(math.isfinite(v) and v > 0 for v in (max_age, max_skew, state_live_age)):
            raise ValueError("Timing limits must be positive and finite")
        self.max_age, self.max_skew, self.state_live_age = max_age, max_skew, state_live_age
        self.buffers = {key: deque(maxlen=24 if key in IMAGE_KEYS else 200) for key in SOURCES}
        self.lock = threading.Lock()
        self.clock_offset = None
        self.accepted = dict.fromkeys(SOURCES, 0)
        self.invalid = dict.fromkeys(SOURCES, 0)

    def add(self, record, payload=b"", *, now_wall=None, now_mono=None):
        now_wall = time.time() if now_wall is None else now_wall
        now_mono = time.monotonic() if now_mono is None else now_mono
        source = record["source"]
        if source not in self.buffers:
            raise ValueError(f"Unknown source: {source}")
        with self.lock:
            try:
                offset = now_wall - now_mono
                if self.clock_offset is not None and abs(offset - self.clock_offset) > 0.05:
                    for history in self.buffers.values():
                        history.clear()
                    self.clock_offset = offset
                    raise ValueError("Host clock jumped; all histories discarded")
                self.clock_offset = offset
                stamp = float(record["stamp"])
                if not math.isfinite(stamp) or not -0.02 <= now_wall - stamp <= 0.5:
                    raise ValueError(
                        f"{source}: invalid, future or excessively old source timestamp (age={now_wall - stamp:.6f}s)"
                    )
                if self.buffers[source] and stamp <= self.buffers[source][-1].stamp:
                    raise ValueError(f"{source}: repeated or backwards timestamp")
                if source in IMAGE_KEYS:
                    shape = IMAGE_SHAPES[source]
                    if record["encoding"] != "rgb8" or tuple(record["shape"]) != shape:
                        raise ValueError(f"{source}: expected RGB uint8 {shape}")
                    if len(payload) != math.prod(shape):
                        raise ValueError(f"{source}: wrong image byte count")
                    value = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
                else:
                    if record.get("units") != "radian" or record.get("measured") is not True or payload:
                        raise ValueError(f"{source}: measured radians required")
                    value = positions_in_order(source, record["names"], record["positions"])
                self.buffers[source].append(
                    Sample(stamp, now_mono, value, record["stamp_basis"], record.get("producer_received_wall"))
                )
                self.accepted[source] += 1
            except (KeyError, TypeError, ValueError):
                self.buffers[source].clear()
                self.invalid[source] += 1
                raise

    def snapshot(self, *, now_wall=None, now_mono=None):
        now_wall = time.time() if now_wall is None else now_wall
        now_mono = time.monotonic() if now_mono is None else now_mono
        with self.lock:
            if self.clock_offset is not None and abs(now_wall - now_mono - self.clock_offset) > 0.05:
                raise ObservationUnavailableError("clock_jump")
            missing = [key for key, history in self.buffers.items() if not history]
            if missing:
                raise ObservationUnavailableError("missing:" + ",".join(missing))
            for key, history in self.buffers.items():
                limit = self.max_age if key in IMAGE_KEYS else self.state_live_age
                if now_mono - history[-1].received > limit or now_wall - history[-1].stamp > limit:
                    raise ObservationUnavailableError("stale:" + key)
            target = min(self.buffers[key][-1].stamp for key in IMAGE_KEYS)
            selected = {
                key: min(history, key=lambda sample: abs(sample.stamp - target))
                for key, history in self.buffers.items()
            }
            stamps = [sample.stamp for sample in selected.values()]
            skew = max(stamps) - min(stamps)
            if skew > self.max_skew:
                raise ObservationUnavailableError("timestamp_skew")
            if any(not -0.02 <= now_wall - stamp <= self.max_age for stamp in stamps):
                raise ObservationUnavailableError("selected_sample_age")
            obs = {"observation/state": np.concatenate([selected[key].value for key in STATE_ORDER]), "prompt": PROMPT}
            obs.update({key: selected[source].value for source, key in IMAGE_KEYS.items()})
            metadata = {
                "target_stamp": target,
                "skew_seconds": skew,
                "samples": {
                    key: {
                        "stamp": sample.stamp,
                        "age_seconds": now_wall - sample.stamp,
                        "stamp_basis": sample.basis,
                        "buffer_wait_seconds": now_mono - sample.received,
                        "age_at_buffer_receipt_seconds": now_wall - sample.stamp - (now_mono - sample.received),
                        "producer_received_wall": sample.producer_received,
                    }
                    for key, sample in selected.items()
                },
                "latest": {
                    key: {
                        "stamp": history[-1].stamp,
                        "age_seconds": now_wall - history[-1].stamp,
                        "buffer_wait_seconds": now_mono - history[-1].received,
                        "age_at_buffer_receipt_seconds": now_wall
                        - history[-1].stamp
                        - (now_mono - history[-1].received),
                        "producer_received_wall": history[-1].producer_received,
                    }
                    for key, history in self.buffers.items()
                },
            }
            return obs, metadata


def send_record(sock, record, payload=b"", *, lock=None):
    def send():
        header = json.dumps({**record, "payload_bytes": len(payload)}, allow_nan=False).encode()
        sock.sendall(struct.pack("!I", len(header)) + header + payload)

    if lock is None:
        send()
    else:
        with lock:
            send()


def read_record(stream):
    def exact(size):
        parts = bytearray()
        while len(parts) < size:
            chunk = stream.read(size - len(parts))
            if not chunk:
                raise EOFError("Observation producer disconnected")
            parts.extend(chunk)
        return bytes(parts)

    size = struct.unpack("!I", exact(4))[0]
    if not 0 < size <= 16384:
        raise ValueError("Invalid observation header length")
    record = json.loads(exact(size))
    count = record["payload_bytes"]
    if type(count) is not int or not 0 <= count <= 2_000_000:
        raise ValueError("Invalid observation payload length")
    return record, exact(count)
