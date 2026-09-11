"""One in-flight inference in an owned process; never imports a hardware writer."""

import contextlib
import multiprocessing as mp
import queue
import time

from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from executor.protocol import Candidate


def request_timing(request, wall, mono):
    """Preserve the capture clock conversion across IPC and serialization."""
    if abs((wall - request["capture_wall"]) - (mono - request["capture_mono"])) > 0.05:
        return "clock_jump", None
    ages = {key: wall - sample["stamp"] for key, sample in request["metadata"]["samples"].items()}
    if not ages or min(ages.values()) < -0.02 or max(ages.values()) > 0.07:
        return "input_exceeds_send_budget", ages
    return None, ages


def _worker(uri, timeout, requests, responses, startup):
    try:
        with connect(uri, compression=None, max_size=None, open_timeout=10, close_timeout=1, proxy=None) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=timeout))
            expected = {
                "config": "pi05_fr3_wuji",
                "action_dim": 54,
                "action_horizon": 50,
                "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
                "hardware_output": False,
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Unexpected inference contract: {metadata}")
            startup.send({"ready": True, "metadata": metadata})
            packer = msgpack_numpy.Packer()
            while True:
                request = requests.get()
                if request is None:
                    return
                started = time.monotonic()
                payload = packer.pack(request["observation"])
                wall, sent = time.time(), time.monotonic()
                problem, ages = request_timing(request, wall, sent)
                common = {"request_id": request["request_id"]}
                if problem:
                    responses.put({**common, "kind": "skipped", "reason": problem})
                    continue
                ws.send(payload)
                payload = ws.recv(timeout=timeout)
                received, returned_wall = time.monotonic(), time.time()
                if abs((returned_wall - wall) - (received - sent)) > 0.05:
                    responses.put({**common, "kind": "skipped", "reason": "clock_jump"})
                    continue
                if isinstance(payload, str):
                    raise RuntimeError(f"Inference server error: {payload}")
                result = msgpack_numpy.unpackb(payload)
                candidate = Candidate(
                    request["run_id"],
                    request["generation"],
                    request["request_id"],
                    str(request["metadata"]["target_stamp"]),
                    sent,
                    received,
                    request["valid_until_mono"],
                    result["actions"],
                    request["observation"]["observation/state"],
                    request["clock_epoch"],
                )
                responses.put(
                    {
                        **common,
                        "kind": "prediction",
                        "candidate": candidate,
                        "timing": {
                            "roundtrip_ms": (received - sent) * 1000,
                            "ipc_and_wait_before_pack_ms": (started - request["capture_mono"]) * 1000,
                            "serialization_ms": (sent - started) * 1000,
                            "input_age_at_send_ms": max(ages.values()) * 1000,
                            "server_timing": result.get("server_timing", {}),
                        },
                    }
                )
    except BaseException as error:
        failure = {"kind": "error", "error": f"{type(error).__name__}: {error}"}
        try:
            startup.send(failure)
            responses.put(failure, timeout=0.1)
        except (OSError, queue.Full):
            pass
    finally:
        startup.close()


class InferenceWorker:
    def __init__(self, uri, *, timeout=2):
        context = mp.get_context("spawn")
        self.requests, self.responses = context.Queue(1), context.Queue(1)
        self.startup, child = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_worker, args=(uri, timeout, self.requests, self.responses, child), daemon=True
        )
        self.pending = False
        self.closed = False
        self.process.start()
        child.close()
        try:
            if not self.startup.poll(12 + timeout):
                raise RuntimeError("Inference worker startup timed out")
            result = self.startup.recv()
            if not result.get("ready"):
                raise RuntimeError(str(result))
            self.metadata = result["metadata"]
        except BaseException:
            self.close()
            raise

    def submit(self, request):
        if self.closed or not self.process.is_alive():
            raise RuntimeError("Inference worker is unavailable")
        if self.pending:
            return False
        self.requests.put_nowait(request)
        self.pending = True
        return True

    def poll(self):
        if self.closed:
            raise RuntimeError("Inference worker is closed")
        try:
            result = self.responses.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise RuntimeError("Inference worker exited") from None
            return None
        self.pending = False
        if result["kind"] == "error":
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(queue.Full):
            self.requests.put_nowait(None)
        self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        self.startup.close()
        for channel in (self.requests, self.responses):
            channel.cancel_join_thread()
            channel.close()
