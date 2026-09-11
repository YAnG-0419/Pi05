"""Local, message-bounded transport. Host and container share CLOCK_MONOTONIC."""

from dataclasses import asdict
import json
import socket
import threading
import time

import numpy as np

from .core import Feedback
from .core import Frame

MAX_PACKET = 64 * 1024


def encode(value):
    packet = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(packet) >= MAX_PACKET:
        raise ValueError("IPC packet is too large")
    return packet


def receive(sock):
    packet, _, flags, _ = sock.recvmsg(MAX_PACKET)
    if not packet:
        raise EOFError("Device IPC disconnected")
    if flags & socket.MSG_TRUNC:
        raise ValueError("Truncated IPC packet")
    return json.loads(packet, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def wire_feedback(feedback):
    return {**asdict(feedback), "positions": feedback.positions.tolist(), "velocities": feedback.velocities.tolist()}


def wire_frame(frame):
    return {**asdict(frame), "positions": frame.positions.tolist(), "velocities": frame.velocities.tolist()}


class RemoteDevices:
    """Synchronous acknowledgements; no queued or retried motion commands."""

    def __init__(self, path, *, execute=False, finish_policy="hold", sock=None):
        self.sock = sock or socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        if sock is None:
            self.sock.settimeout(5)
            self.sock.connect(str(path))
        self.counter = 0
        self.hardware_output = execute
        self.finish_policy = finish_policy
        self.closed = False
        self.stop_result = None
        self.last_submission = None
        self.lock = threading.Lock()
        self.hold_stopping = threading.Event()
        self.hold_thread = None
        self.hold_error = None

    def call(self, operation, *, timeout=0.15, **values):
        with self.lock:
            return self._call(operation, timeout=timeout, **values)

    def _call(self, operation, *, timeout, **values):
        self.counter += 1
        self.sock.settimeout(timeout)
        request = {"id": self.counter, "operation": operation, **values}
        self.sock.sendall(encode(request))
        result = receive(self.sock)
        if result.get("id") != self.counter:
            raise RuntimeError("IPC reply sequence mismatch; do not retry motion")
        if result.get("ok") is not True:
            raise RuntimeError(result.get("error", "Device bridge rejected operation"))
        return result["result"]

    def feedback(self, now):
        result = self.call("feedback")
        feedback = Feedback(**result)
        feedback.check(time.monotonic(), feedback.epoch)
        return feedback

    def prepare(self, player, feedback, now):
        if not self.hardware_output:
            raise RuntimeError("Read-only client cannot acquire devices")
        self.call(
            "prepare",
            timeout=6,
            run_id=player.run_id,
            plan_hash=player.digest,
            start=player.plan.start.tolist(),
            finish_policy=self.finish_policy,
        )

    def submit(self, frame, now):
        remaining = frame.valid_until - time.monotonic()
        if remaining <= 0:
            raise ValueError("Frame expired before IPC send")
        result = self.call("submit", timeout=remaining, frame=wire_frame(frame))
        if result.get("sequence") != frame.sequence:
            raise RuntimeError("Wrong submitted-frame acknowledgement")
        self.last_submission = result

    def finish(self):
        result = self.call("finish", timeout=3)
        if result["state"] == "holding":

            def monitor():
                try:
                    while not self.hold_stopping.wait(0.05):
                        self.monitor_hold()
                except Exception as error:
                    self.hold_error = str(error)
                    self.close()

            self.hold_thread = threading.Thread(target=monitor, daemon=True)
            self.hold_thread.start()
        return result

    def monitor_hold(self):
        return self.call("monitor")

    def stop(self):
        self.hold_stopping.set()
        if self.hold_thread is not None and self.hold_thread is not threading.current_thread():
            self.hold_thread.join(timeout=0.3)
        if self.closed or self.stop_result is not None:
            return self.stop_result
        try:
            self.stop_result = self.call("stop", timeout=2)
            return self.stop_result
        except (OSError, EOFError, ValueError, RuntimeError):
            # A timed-out request may still have a reply queued. Disconnecting
            # makes the device process stop independently; never report success.
            self.close()
            raise

    def close(self):
        self.hold_stopping.set()
        if not self.closed:
            self.closed = True
            self.sock.close()


def unpack_frame(value):
    frame = Frame(**value)
    if type(frame.sequence) is not int or frame.sequence >= 2**64:
        raise ValueError("Invalid frame sequence")
    if not np.isfinite(frame.positions).all():
        raise ValueError("Invalid frame positions")
    return frame
