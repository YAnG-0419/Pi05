"""GC restoration, stop ordering and allocation stress without hardware."""

import gc

import pytest

from experiments.weight_motion_eval.oneshot.motion_gc import MotionGC
from experiments.weight_motion_eval.oneshot.test_devices import frame
from experiments.weight_motion_eval.oneshot.test_devices import make_session
from experiments.weight_motion_eval.oneshot.test_devices import prepare


@pytest.mark.parametrize("initially_enabled", [False, True])
def test_gc_restores_previous_policy(initially_enabled):
    original = gc.isenabled()
    policy = MotionGC()
    try:
        (gc.enable if initially_enabled else gc.disable)()
        policy.begin()
        assert not gc.isenabled()
        policy.end()
        policy.end()
        assert gc.isenabled() == initially_enabled
    finally:
        (gc.enable if original else gc.disable)()


@pytest.mark.parametrize("enable_failure", [False, True])
def test_owner_stops_before_restoring_gc(enable_failure):
    original = gc.isenabled()
    session = make_session()
    session.motion_gc = MotionGC()
    stop_states = []
    original_stop = session.arms.stop
    def stop():
        stop_states.append(gc.isenabled())
        original_stop()
    session.arms.stop = stop
    if enable_failure:
        def fail(**kwargs):
            raise RuntimeError("injected enable failure")
        session.hands["right"].enable = fail
    try:
        gc.enable()
        if enable_failure:
            with pytest.raises(RuntimeError, match="injected"):
                prepare(session)
        else:
            prepare(session)
            assert not gc.isenabled()
            session.submit(frame())
            session.stop()
        assert stop_states == [False]
        assert gc.isenabled()
        assert not session.motion_gc.active
    finally:
        session.stop()
        (gc.enable if original else gc.disable)()


def test_trace_allocation_stress_has_no_automatic_collections():
    from types import SimpleNamespace

    from experiments.weight_motion_eval.oneshot.hand_trace import HandTrace
    policy = MotionGC()
    trace = HandTrace(capacity=100)
    sdk_frame = SimpleNamespace(header=SimpleNamespace(timestamp_us=1), num_joints=20,
                                joints=[SimpleNamespace(nid=i, position=0, velocity=0) for i in range(20)])
    try:
        policy.begin()
        before = [row["collections"] for row in gc.get_stats()]
        for _ in range(20000):
            trace.frame(sdk_frame)
        assert [row["collections"] for row in gc.get_stats()] == before
        assert len(trace.before) == 100
    finally:
        policy.end()
