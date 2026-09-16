"""Directional position caps for diagnosed slider stalls; not force control."""

from dataclasses import dataclass

import numpy as np

DIRECTION_EPS_RAD = 0.005
RETREAT_RELEASE_RAD = 0.02


@dataclass(frozen=True)
class Contact:
    direction: int
    bound: float
    measured_at_trigger: float
    triggered_at: float


class StallSoftLimit:
    def __init__(self):
        # At most the 20 physical joints; survives prediction/round changes.
        self.contacts = {}

    def project(self, target):
        result = np.asarray(target, dtype=float).copy()
        for index, contact in self.contacts.items():
            if contact.direction * (result[index] - contact.bound) > 0:
                result[index] = contact.bound
        return result

    def update(self, target, measured, last_command, stalled, now):
        events = []
        for index in sorted(stalled):
            if index in self.contacts:
                continue
            # Position error of the command actually sent indicates the loading
            # direction. Do not assume all fingers close towards positive angles.
            error = float(last_command[index] - measured[index])
            if abs(error) <= DIRECTION_EPS_RAD:
                raise ValueError(f"Stalled hand joint index={index}: loading direction is ambiguous")
            contact = Contact(1 if error > 0 else -1, float(last_command[index]), float(measured[index]), now)
            self.contacts[index] = contact
            events.append({"action": "engaged", "index": index, **contact.__dict__})
        for index, contact in list(self.contacts.items()):
            # Warning clearing alone never reopens the blocked direction.
            if (index not in stalled
                    and contact.direction * (target[index] - contact.bound) <= -RETREAT_RELEASE_RAD
                    and contact.direction * (last_command[index] - contact.bound) <= -RETREAT_RELEASE_RAD
                    and contact.direction * (measured[index] - contact.measured_at_trigger) <= -RETREAT_RELEASE_RAD):
                del self.contacts[index]
                events.append({"action": "released", "index": index, **contact.__dict__})
        return self.project(target), events

    def snapshot(self, offset):
        return tuple((offset + index, contact.direction, contact.bound) for index, contact in sorted(self.contacts.items()))
