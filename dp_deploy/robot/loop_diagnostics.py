"""Bounded in-memory loop diagnostics plus an optional separate CPU sampler."""

from collections import deque
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time


class LoopDiagnostics:
    def __init__(self):
        self.events = deque(maxlen=512)
        self.stage = "startup"
        self.gc_start = None
        self.gc_counts = [0, 0, 0]
        self.gc_max_ms = [0.0, 0.0, 0.0]
        self.dropped = 0
        self.last_loop = None

    def add(self, event):
        self.dropped += len(self.events) == self.events.maxlen
        self.events.append(event)

    def collection(self, phase, info):
        now = time.monotonic()
        if phase == "start":
            self.gc_start = now
        elif self.gc_start is not None:
            elapsed = (now - self.gc_start) * 1000
            generation = info["generation"]
            self.gc_counts[generation] += 1
            self.gc_max_ms[generation] = max(self.gc_max_ms[generation], elapsed)
            if elapsed >= 1:
                self.add({"event": "gc", "at": now, "stage": self.stage, "generation": generation, "elapsed_ms": elapsed})
            self.gc_start = None

    def loop(self):
        now = time.monotonic()
        if self.last_loop is not None and now - self.last_loop >= 0.005:
            self.add({"event": "loop_gap", "at": now, "elapsed_ms": (now - self.last_loop) * 1000})
        self.last_loop = now

    def start(self):
        gc.callbacks.append(self.collection)

    @contextmanager
    def measure(self, stage, threshold_ms=2):
        previous = self.stage
        self.stage = stage
        began, cpu = time.monotonic(), time.thread_time()
        try:
            yield
        finally:
            now = time.monotonic()
            elapsed = (now - began) * 1000
            if elapsed >= threshold_ms:
                self.add({"event": "slow_stage", "at": now, "stage": stage, "elapsed_ms": elapsed,
                              "cpu_ms": (time.thread_time() - cpu) * 1000})
            self.stage = previous

    def close(self, path):
        gc.callbacks.remove(self.collection)
        Path(path).write_text(json.dumps({"events": list(self.events), "dropped": self.dropped,
                                             "gc_counts": self.gc_counts, "gc_max_ms": self.gc_max_ms}, indent=2) + "\n")


def sample_cpu(pid, output):
    """Separate process: 1 Hz /proc reads and disk writes never run on owner thread."""
    import os
    previous = None
    with Path(output).open("x") as stream:
        while Path(f"/proc/{pid}").exists():
            began = time.monotonic()
            cpus = {}
            for line in Path("/proc/stat").read_text().splitlines():
                fields = line.split()
                if fields and fields[0].startswith("cpu"):
                    ticks = list(map(int, fields[1:9]))
                    cpus[fields[0]] = (sum(ticks), ticks[3] + ticks[4])
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                break
            process_ticks = int(fields[11]) + int(fields[12])
            memory = {}
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    key, _, value = line.partition(":")
                    if key in {"VmRSS", "VmHWM", "VmSize"}:
                        memory[key + "_kb"] = int(value.split()[0])
            except FileNotFoundError:
                break
            event = {"at": began, "loadavg": os.getloadavg(), "cpu_percent": {}, "process_cpu_percent": None, "memory": memory}
            if previous:
                old_time, old_cpus, old_process = previous
                for cpu, (total, idle) in cpus.items():
                    if cpu in old_cpus and total > old_cpus[cpu][0]:
                        event["cpu_percent"][cpu] = 100 * (1 - (idle - old_cpus[cpu][1]) / (total - old_cpus[cpu][0]))
                event["process_cpu_percent"] = 100 * (process_ticks - old_process) / os.sysconf("SC_CLK_TCK") / (began - old_time)
            event["sample_ms"] = (time.monotonic() - began) * 1000
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            previous = began, cpus, process_ticks
            time.sleep(max(0, 1 - (time.monotonic() - began)))


if __name__ == "__main__":
    import sys
    sample_cpu(int(sys.argv[1]), sys.argv[2])
