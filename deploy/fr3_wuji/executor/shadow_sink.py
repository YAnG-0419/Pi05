"""In-memory sink with the same final-consumption guards as an output adapter."""

from collections import deque

from executor.protocol import finite


class ShadowSink:
    hardware_output = False

    def __init__(self, run_id, generation=0, *, capacity=256):
        self.run_id, self.generation = run_id, generation
        self.last_frame = -1
        self.frames = deque(maxlen=capacity)
        self.submitted = 0
        self.stop_count = 0

    def fence(self, run_id, generation):
        self.run_id, self.generation = run_id, generation
        self.last_frame = -1

    def submit(self, frame, now):
        finite(now)
        if (frame.run_id, frame.generation) != (self.run_id, self.generation):
            raise ValueError("stale_generation")
        if frame.frame_id <= self.last_frame:
            raise ValueError("duplicate_frame")
        if not frame.created_mono <= now < frame.valid_until_mono:
            raise ValueError("expired_or_future_frame")
        self.frames.append(frame)
        self.last_frame = frame.frame_id
        self.submitted += 1

    def stop(self):
        self.stop_count += 1
        return True  # Only an in-memory sink; never a claim of physical stopping.
