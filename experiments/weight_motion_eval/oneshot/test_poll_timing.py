"""Stage attribution and callback cleanup, without SDK or physical output."""

import gc

import pytest

from .poll_timing import PollTiming


def test_gc_is_attributed_and_callback_removed_on_exception():
    before = list(gc.callbacks)
    timing = PollTiming()
    def failing_poll():
        with timing:
            timing.mark("trace_ms")
            gc.collect(2)
            raise RuntimeError("injected")
    with pytest.raises(RuntimeError, match="injected"):
        failing_poll()
    assert gc.callbacks == before
    assert timing.result["gc_count_by_generation"][2] >= 1
    assert timing.result["trace_ms"] >= timing.result["gc_ms_by_generation"][2]
    assert timing.result["total_ms"] >= timing.result["trace_ms"]


def test_stage_intervals_accumulate_without_double_counting(monkeypatch):
    from . import poll_timing
    now = [10.0]
    monkeypatch.setattr(poll_timing.time, "perf_counter", lambda: now[0])
    with PollTiming() as timing:
        timing.mark("state_recv_ms")
        now[0] += 0.003
        timing.mark("trace_ms")
        now[0] += 0.030
        timing.mark("state_recv_ms")
        now[0] += 0.001
    assert timing.result["state_recv_ms"] == pytest.approx(4)
    assert timing.result["trace_ms"] == pytest.approx(30)
    assert timing.result["total_ms"] == pytest.approx(34)
