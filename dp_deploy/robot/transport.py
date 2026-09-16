"""Preserve per-frame deadlines through existing ROS message headers."""

import math
import re

PREFIX = "pi05v1"
PATTERN = re.compile(r"pi05v1/([0-9a-f]{32})/([0-9]+)/([0-9]+)\Z")


def ros_header(frame, monotonic_now, wall_now_ns):
    if not all(math.isfinite(x) for x in (monotonic_now, frame.created, frame.valid_until)):
        raise ValueError("Invalid monotonic time")
    if not frame.created <= monotonic_now < frame.valid_until:
        raise ValueError("Frame expired before ROS serialization")
    created_ns = wall_now_ns - round((monotonic_now - frame.created) * 1e9)
    expires_ns = wall_now_ns + round((frame.valid_until - monotonic_now) * 1e9)
    tag = f"{PREFIX}/{frame.run_id}/{frame.sequence}/{expires_ns}"
    validate_header(tag, created_ns, wall_now_ns)
    return created_ns, tag


def validate_header(tag, created_ns, now_ns):
    match = PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError("Missing or malformed one-shot command identity/deadline")
    run, sequence, expiry = match.groups()
    sequence, expiry = int(sequence), int(expiry)
    if not 0 <= sequence < 2**64 or not 0 < expiry < 2**63:
        raise ValueError("Command identity exceeds transport range")
    if not 0 < created_ns < expiry or not 0 < expiry - created_ns <= 20_000_001:
        raise ValueError("ROS frame lifetime exceeds 20 ms")
    if not created_ns <= now_ns < expiry:
        raise ValueError("Expired or future command at ROS consumer")
    return run, sequence, expiry
