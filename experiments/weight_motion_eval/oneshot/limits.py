"""Shared per-joint hand command and measured-speed ceiling for this trial."""

import math

HAND_SPEED_DEG_S = 45.0
HAND_SPEED_RAD_S = math.radians(HAND_SPEED_DEG_S)
HAND_KP = 8.0
HAND_KD = 0.1
HAND_CURRENT_A = 1.0
SLIDER_CURRENT_A = 2.0
HAND_CONTACT_SECONDS = 300.0
HAND_CONTACT_SETTLE_SECONDS = 20.0
# Runtime captures show 1.016 ms clock lead; allow bounded 2 ms jitter.
HAND_CLOCK_LEAD_SECONDS = 0.002
ARM_TRACKING_TOLERANCE_RAD = 0.08
SLIDER_SPEED_RAD_S = 1.0
SLIDER_POSITION_TOLERANCE_RAD = 0.25
HAND_ENDPOINT_TOLERANCE_RAD = 0.25
ARM_ENDPOINT_TOLERANCE_RAD = 0.25


def check_hand_control(value):
    if value not in {"strict", "slider"}:
        raise ValueError("Hand control must be strict or slider")
    return value
