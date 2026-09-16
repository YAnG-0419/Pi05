"""Time-bounded slider grasping after a diagnosed Stall warning.

The firmware current ceiling and SDK position/slew checks still apply. Contact
does not project the model target: it supplies a timed endpoint alternative.
"""

import math

from .hand_soft_limit import StallSoftLimit
from .limits import HAND_CONTACT_SECONDS


class BoundedContactGrasp(StallSoftLimit):
    def project(self, target):
        return target.copy()

    def check_time(self, now):
        for index, contact in self.contacts.items():
            age = now - contact.triggered_at
            if not math.isfinite(age) or not 0 <= age < HAND_CONTACT_SECONDS:
                nid = index // 4 * 5 + index % 4 + 1
                raise ValueError(
                    f"Hand contact duration exceeded: NID={nid}, "
                    f"elapsed={age:.3f}s, limit={HAND_CONTACT_SECONDS:g}s"
                )

    def update(self, target, measured, last_command, stalled, now):
        # Inherited release requires warning clearance AND actual retreat;
        # neither a new prediction nor warning flicker restarts this timer.
        self.check_time(now)
        return super().update(target, measured, last_command, stalled, now)

    def snapshot(self, offset=0):
        return tuple((offset + index, c.direction, c.bound, c.triggered_at)
                     for index, c in sorted(self.contacts.items()))
