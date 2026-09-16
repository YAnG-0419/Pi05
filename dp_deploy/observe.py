"""Read-only ROS observation collector with causal, 30 Hz, two-frame windows."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import numpy as np
from .config import load_config
from .contract import DP_GROUPS, RGB_KEYS, ordered_positions
from .wire import pack


class ObservationUnavailable(RuntimeError):
    pass


class Buffer:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.buffers = {
            k: deque(maxlen=120) for k in (*DP_GROUPS, *RGB_KEYS, "head_depth")
        }
        self.offset = time.time() - time.monotonic()
        self.errors = {}

    def add(self, key, stamp, value, received=None):
        received = time.monotonic() if received is None else received
        if not np.isfinite(stamp) or stamp <= 0:
            raise ValueError("Invalid ROS timestamp")
        with self.lock:
            q = self.buffers[key]
            if q and stamp < q[-1][0]:
                for history in self.buffers.values():
                    history.clear()
                raise ValueError("ROS time moved backwards; history cleared")
            if q and stamp == q[-1][0]:
                return
            q.append((stamp, received, value))
            self.errors.pop(key, None)

    def snapshot(self, wall=None, mono=None):
        wall = time.time() if wall is None else wall
        mono = time.monotonic() if mono is None else mono
        if abs(wall - mono - self.offset) > 0.05:
            raise ObservationUnavailable("Host clock jumped")
        with self.lock:
            missing = [k for k, v in self.buffers.items() if not v]
            if missing:
                raise ObservationUnavailable("Missing: " + ",".join(missing))
            # Resample causal sources on the training 30 Hz clock; camera rates may differ.
            target = np.floor(min(v[-1][0] for v in self.buffers.values()) * 30) / 30
            targets = np.array([target - 1 / 30, target])
            if not 0 <= wall - target <= self.config["max_observation_age_s"]:
                raise ObservationUnavailable("Stale target")
            selected = {}
            source_stamps = []
            for key, history in self.buffers.items():
                max_age = self.config[
                    "max_state_age_s" if key in DP_GROUPS else "max_image_age_s"
                ]
                latest = history[-1]
                if not 0 <= mono - latest[1] <= self.config["max_observation_age_s"]:
                    raise ObservationUnavailable("Stale receipt: " + key)
                frames = []
                stamps = []
                for t in targets:
                    candidates = [s for s in history if s[0] <= t + 1e-7]
                    if not candidates:
                        raise ObservationUnavailable(
                            "Need two historical frames: " + key
                        )
                    s = candidates[-1]
                    if not -1e-7 <= t - s[0] <= max_age:
                        raise ObservationUnavailable("Source gap: " + key)
                    frames.append(s[2])
                    stamps.append(s[0])
                selected[key] = np.stack(frames)
                source_stamps.append(stamps)
            result = {k: selected[k] for k in (*RGB_KEYS, "head_depth")}
            result["joint_pos"] = np.concatenate(
                [selected[k] for k in DP_GROUPS], axis=-1
            )
            result["timestamps"] = targets
            result["source_timestamps"] = np.asarray(source_stamps)
            result["captured_wall"] = np.array(wall)
            return result


def decode_image(msg, depth=False):
    encoding = msg.encoding.lower()
    h, w = int(msg.height), int(msg.width)
    if h <= 0 or w <= 0 or h * w > 4096 * 4096:
        raise ValueError("Invalid image dimensions")
    if depth:
        if encoding not in ("16uc1", "32fc1"):
            raise ValueError("Depth must be 16UC1 or 32FC1")
        dtype = np.dtype(
            (">" if msg.is_bigendian else "<") + ("u2" if encoding == "16uc1" else "f4")
        )
        channels = 1
    else:
        if encoding not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            raise ValueError("Unsupported RGB encoding: " + encoding)
        dtype = np.dtype("u1")
        channels = 4 if "a8" in encoding else 3
    row = w * channels * dtype.itemsize
    if msg.step < row or len(msg.data) != h * msg.step:
        raise ValueError("Invalid image row stride")
    raw = np.ndarray(
        (h, w, channels),
        dtype=dtype,
        buffer=bytes(msg.data),
        strides=(msg.step, channels * dtype.itemsize, dtype.itemsize),
    )
    if depth:
        return raw[..., 0].astype(np.uint16 if encoding == "16uc1" else np.float32)
    raw = raw[..., :3]
    if encoding.startswith("bgr"):
        raw = raw[..., ::-1]
    return raw.copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8876)
    a = p.parse_args()
    c = load_config()
    store = Buffer(c)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    rclpy.init()
    node = rclpy.create_node("dp_readonly_observation")

    def callback(key, msg):
        try:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            value = (
                ordered_positions(key, msg.name, msg.position)
                if key in DP_GROUPS
                else decode_image(msg, key == "head_depth")
            )
            store.add(key, stamp, value)
        except Exception as e:
            with store.lock:
                store.errors[key] = str(e)

    for key, topic in c["topics"].items():
        node.create_subscription(
            JointState if key in DP_GROUPS else Image,
            topic,
            lambda msg, k=key: callback(k, msg),
            qos_profile_sensor_data,
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                if self.path != "/observation":
                    raise ValueError("Unknown route")
                result = pack(store.snapshot())
                code = 200
                kind = "application/octet-stream"
            except Exception as e:
                code = 503
                kind = "application/json"
                result = json.dumps(
                    {"error": str(e), "invalid_sources": store.errors}
                ).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Read-only observation port {a.port}", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
