"""Diagnostics run without devices and cannot erase the first stop reason."""

import gc
import json
import os
import subprocess
import sys
import time

import pytest

from experiments.weight_motion_eval.oneshot.loop_diagnostics import LoopDiagnostics


def test_process_gc_and_failed_stage_are_retained_and_bounded(tmp_path):
    diagnostics = LoopDiagnostics()
    previous = list(gc.callbacks)
    diagnostics.start()
    def fail():
        with diagnostics.measure("ros_spin", threshold_ms=0):
            gc.collect(2)
            raise ValueError("injected")
    with pytest.raises(ValueError, match="injected"):
        fail()
    assert diagnostics.gc_counts[2] >= 1
    assert any(e.get("stage") == "ros_spin" for e in diagnostics.events)
    for i in range(600):
        diagnostics.add({"event": "test", "sequence": i})
    path = tmp_path / "loop.json"
    diagnostics.close(path)
    assert gc.callbacks == previous
    result = json.loads(path.read_text())
    assert len(result["events"]) == 512
    assert result["dropped"] >= 88


def test_cpu_sampler_records_host_and_process_load(tmp_path):
    path = tmp_path / "cpu.jsonl"
    process = subprocess.Popen([
        sys.executable, "-m", "experiments.weight_motion_eval.oneshot.loop_diagnostics", str(os.getpid()), str(path)
    ])
    try:
        deadline = time.monotonic() + 5
        rows = []
        while time.monotonic() < deadline:
            if path.exists():
                lines = path.read_text().splitlines()
                rows = [json.loads(line) for line in lines if line.endswith("}")]
                if len(rows) >= 2:
                    break
            time.sleep(0.05)
        assert len(rows) >= 2
        assert "cpu" in rows[-1]["cpu_percent"]
        assert rows[-1]["process_cpu_percent"] >= 0
        assert len(rows[-1]["loadavg"]) == 3
        assert rows[-1]["memory"]["VmRSS_kb"] > 0
    finally:
        process.terminate()
        process.wait(timeout=5)
