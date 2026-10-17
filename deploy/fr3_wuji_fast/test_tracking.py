"""Fast tracking threshold reaches both checks without changing slow defaults."""

from dataclasses import replace
import queue

import numpy as np
import pytest

from deploy.fr3_wuji_fast.control import run_control
from deploy.fr3_wuji_fast.limits import ARM_TRACKING_TOLERANCE_RAD as FAST_TRACKING
from deploy.fr3_wuji_fast.test_fast import LIMITS
from deploy.fr3_wuji_fast.test_fast import Clock
from deploy.fr3_wuji_fast.test_fast import Devices
from deploy.fr3_wuji_fast.test_fast import initial
from experiments.weight_motion_eval.oneshot.core import ARM
from experiments.weight_motion_eval.oneshot.core import ConsumerGuard
from experiments.weight_motion_eval.oneshot.core import Feedback
from experiments.weight_motion_eval.oneshot.core import Frame
from experiments.weight_motion_eval.oneshot.core import check_tracking
from experiments.weight_motion_eval.oneshot.devices import DeviceSession
from experiments.weight_motion_eval.oneshot.limits import ARM_TRACKING_TOLERANCE_RAD as SLOW_TRACKING


@pytest.mark.parametrize("error", [0.080039, 0.19999, 0.2, 0.20001])
@pytest.mark.parametrize("fast", [False, True])
def test_tracking_and_device_guard_boundaries(error, fast):
    assert SLOW_TRACKING == 0.08
    assert FAST_TRACKING == 0.2
    kwargs = {"arm_tracking_rad": FAST_TRACKING} if fast else {}
    session = DeviceSession(None, {}, LIMITS, **kwargs)
    guard = session.guard
    fb = Feedback(np.zeros(54), np.zeros(54), (10.0,) * 4, (10.0,) * 4)
    guard.arm("run", "hash", fb, 10.0)
    q = np.zeros(54)
    q[6] = -error
    fb = replace(fb, positions=q)
    frame = Frame("run", "hash", 0, 10.0, 10.02, 0, np.zeros(54), np.zeros(54), "playback")
    checks = [
        lambda: check_tracking(frame.positions, fb, ARM, "tracking", **kwargs),
        lambda: guard.validate(frame, fb, 10.0),
    ]
    for check in checks:
        if error <= (FAST_TRACKING if fast else SLOW_TRACKING):
            check()
        else:
            with pytest.raises(ValueError, match="error exceeds"):
                check()


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), True])
def test_invalid_tracking_configuration(value):
    with pytest.raises(ValueError, match="tracking tolerance"):
        ConsumerGuard(LIMITS["lower"], LIMITS["upper"], arm_tracking_rad=value)


@pytest.mark.parametrize("error", [0.1, 0.20001])
def test_fast_control_uses_its_own_tracking_threshold(error):
    clock, events = Clock(), []
    devices = Devices(clock)
    devices.guard = ConsumerGuard(
        LIMITS["lower"],
        LIMITS["upper"],
        hand_control="slider",
        arm_speed_enforced=False,
        arm_speed_rad_s=None,
        arm_tracking_rad=FAST_TRACKING,
    )
    submit = devices.submit

    def lag_after_first_playback_frame(frame, now):
        fb = submit(frame, now)
        if frame.phase == "playback":
            q = fb.positions.copy()
            q[6] = frame.positions[6] - error
            return replace(fb, positions=q)
        return fb

    devices.submit = lag_after_first_playback_frame

    def run():
        run_control(
            devices,
            initial(),
            LIMITS,
            {"rounds": 1},
            queue.Queue(),
            events.append,
            lambda *args: None,
            lambda: False,
            clock=clock,
            sleep=clock.sleep,
        )

    if error < FAST_TRACKING:
        run()
        assert devices.finishes == 1
    else:
        with pytest.raises(ValueError, match="Normal-speed tracking error exceeds 0.20 rad"):
            run()
        assert devices.finishes == 0
