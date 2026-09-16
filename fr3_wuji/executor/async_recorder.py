"""Bounded nonblocking logging with disk I/O in a separate spawned process."""

import json
import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np


class RecordingError(RuntimeError):
    pass


def _writer(output, events, artifacts, control, status, write_delay):
    counts = {"events": 0, "artifacts": 0}
    target = None
    try:
        with (Path(output) / "events.jsonl").open("x") as stream:
            status.send({"ready": True})
            while True:
                if control.poll():
                    target = control.recv()
                work = False
                for _ in range(32):
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        break
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                    counts["events"] += 1
                    work = True
                try:
                    index, values = artifacts.get_nowait()
                except queue.Empty:
                    pass
                else:
                    if write_delay:
                        time.sleep(write_delay)
                    np.savez_compressed(Path(output) / f"sample-{index:06d}.npz", **values)
                    counts["artifacts"] += 1
                    work = True
                if target is not None and all(counts[key] == target[key] for key in counts):
                    stream.flush()
                    status.send({"complete": True, **counts})
                    return
                if not work:
                    stream.flush()
                    time.sleep(0.002)
    except BaseException as error:
        status.send({"error": f"{type(error).__name__}: {error}", **counts})
    finally:
        status.close()
        control.close()


class AsyncRecorder:
    def __init__(self, output, *, event_capacity=2048, artifact_capacity=8, _write_delay=0):
        if event_capacity < 1 or artifact_capacity < 1:
            raise ValueError("Queue capacities must be positive")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        context = mp.get_context("spawn")
        self.events = context.Queue(event_capacity)
        self.artifacts = context.Queue(artifact_capacity)
        child_control, self.control = context.Pipe(duplex=False)
        self.status, child_status = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_writer, args=(str(output), self.events, self.artifacts, child_control, child_status, _write_delay)
        )
        self.counts = {"events": 0, "artifacts": 0}
        self.artifacts_dropped = 0
        self.error = None
        self.closed = False
        self.process.start()
        child_control.close()
        child_status.close()
        if not self.status.poll(10):
            self.process.terminate()
            self.process.join(5)
            raise RecordingError("Logger startup timed out")
        result = self.status.recv()
        if result.get("ready") is not True:
            self.process.join(5)
            raise RecordingError(str(result))

    def check(self):
        if self.closed:
            raise RecordingError("Logger is closed")
        if self.status.poll():
            try:
                self.error = str(self.status.recv())
            except EOFError:
                self.error = "Logger disconnected"
        if self.error or not self.process.is_alive():
            raise RecordingError(self.error or "Logger process died")

    def event(self, value):
        self.check()
        # Validate/copy before Queue's feeder thread; serialization failures must
        # reach the scheduler instead of silently losing a critical event.
        snapshot = json.loads(json.dumps(value, allow_nan=False))
        try:
            self.events.put_nowait(snapshot)
        except queue.Full as error:
            self.error = "Critical event queue is full"
            raise RecordingError(self.error) from error
        self.counts["events"] += 1

    def artifact(self, index, values):
        self.check()
        if type(index) is not int or index < 0:
            raise ValueError("Invalid artifact index")
        snapshot = {key: np.array(value, copy=True) for key, value in values.items()}
        if any(value.dtype.hasobject for value in snapshot.values()):
            raise ValueError("Object arrays are not permitted in artifacts")
        try:
            self.artifacts.put_nowait((index, snapshot))
        except queue.Full:
            self.artifacts_dropped += 1
            return False
        self.counts["artifacts"] += 1
        return True

    def close(self, timeout=10):
        if self.closed:
            return None
        self.closed = True
        result = None
        try:
            if self.process.is_alive():
                self.control.send(self.counts)
                if self.status.poll(timeout):
                    result = self.status.recv()
                else:
                    self.error = "Logger did not drain before deadline"
            self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
            if result is None or result.get("complete") is not True:
                raise RecordingError(self.error or str(result))
            return {**result, "artifacts_dropped": self.artifacts_dropped}
        finally:
            self.control.close()
            self.status.close()
            for channel in (self.events, self.artifacts):
                channel.cancel_join_thread()
                channel.close()
