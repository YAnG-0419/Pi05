"""Bounded per-poll timings; GC time overlaps the active stage, never adds to it."""

import gc
import time


class PollTiming:
    def __init__(self):
        self.parts = dict.fromkeys(("state_recv_ms", "diagnostic_recv_ms", "trace_ms", "diagnostic_check_ms", "decode_ms", "other_ms"), 0.0)
        self.counts = {"state_frames": 0, "diagnostic_frames": 0}
        self.gc_ms = [0.0, 0.0, 0.0]
        self.gc_counts = [0, 0, 0]
        self.gc_started = None
        self.stage = "other_ms"

    def collection(self, phase, info):
        if phase == "start":
            self.gc_started = time.perf_counter()
        elif self.gc_started is not None:
            generation = info["generation"]
            self.gc_ms[generation] += (time.perf_counter() - self.gc_started) * 1000
            self.gc_counts[generation] += 1
            self.gc_started = None

    def __enter__(self):
        self.started = self.previous = time.perf_counter()
        self.cpu = time.thread_time()
        gc.callbacks.append(self.collection)
        return self

    def mark(self, stage):
        now = time.perf_counter()
        self.parts[self.stage] += (now - self.previous) * 1000
        self.previous, self.stage = now, stage

    def __exit__(self, *_):
        gc.callbacks.remove(self.collection)
        self.mark("other_ms")
        self.result = {
            **self.parts, **self.counts,
            "total_ms": (time.perf_counter() - self.started) * 1000,
            "cpu_ms": (time.thread_time() - self.cpu) * 1000,
            "gc_ms_by_generation": self.gc_ms,
            "gc_count_by_generation": self.gc_counts,
        }
