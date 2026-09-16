"""Process-local automatic cyclic GC policy for a finite device session."""

import gc
import time


class MotionGC:
    def __init__(self):
        self.active = False
        self.was_enabled = False
        self.report = {}

    def begin(self):
        if self.active:
            return
        self.was_enabled = gc.isenabled()
        gc.disable()
        self.active = True
        began = time.monotonic()
        try:
            collected = gc.collect()
            self.report = {"policy": "defer_automatic_cyclic_gc_until_stop", "preflight_collected": collected,
                           "preflight_ms": (time.monotonic() - began) * 1000, "active": True}
        except BaseException:
            self.end()
            raise

    def end(self):
        if not self.active:
            return
        # Do not collect synchronously: callers must issue physical stop first.
        if self.was_enabled:
            gc.enable()
        self.active = False
        self.report["active"] = False
