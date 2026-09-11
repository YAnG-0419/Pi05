"""Bind commissioning records to installed devices and controller artifacts.

This checks recorded evidence, not physical safety. It never manufactures a
passing record or treats an operator boolean as a device test.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

from .limits import HAND_SPEED_RAD_S


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Qualification:
    def __init__(self, path, *, controller_sha256):
        if path is None:
            raise ValueError("Hardware execution requires measured commissioning evidence (--qualification)")
        path = Path(path).resolve()
        self.data = json.loads(path.read_text())
        if self.data.get("schema_version") != 1 or self.data.get("controller_sha256") != controller_sha256:
            raise ValueError("Commissioning record/controller build mismatch")
        if self.data.get("hand_velocity_rad_s") != HAND_SPEED_RAD_S:
            raise ValueError("Commissioning record must use the agreed hand speed")
        acquisition = self.data.get("hand_raw_initial_delta_rad")
        if (
            isinstance(acquisition, bool)
            or not isinstance(acquisition, int | float)
            or not math.isfinite(acquisition)
            or acquisition <= 0
        ):
            raise ValueError("Commissioning record must specify a measured hand acquisition bound")
        for test in (
            "arm_tracking_and_stop",
            "left_process_loss",
            "left_network_loss",
            "left_enable_no_jump",
            "right_process_loss",
            "right_network_loss",
            "right_enable_no_jump",
        ):
            result = self.data.get("tests", {}).get(test, {})
            evidence_name = result.get("evidence")
            if not isinstance(evidence_name, str) or not evidence_name:
                raise ValueError("Missing measured commissioning evidence: " + test)
            evidence = (path.parent / evidence_name).resolve()
            if (
                not evidence.is_relative_to(path.parent)
                or result.get("passed") is not True
                or not evidence.is_file()
                or digest(evidence) != result.get("sha256")
            ):
                raise ValueError("Missing or changed measured commissioning evidence: " + test)

    def check_hand(self, side, identity):
        if self.data.get("hands", {}).get(side) != identity:
            raise ValueError("Hand identity, firmware or operating parameters differ from commissioning: " + side)


class SupervisedTrial:
    """Explicit single-run waiver of historical evidence, not a passed test.

    Device identity/parameters are bound from fresh readback before acquisition.
    All motion, feedback, ownership and deadline guards remain in force.
    """

    def __init__(self):
        self.data = {
            "execution_mode": "supervised_trial",
            "historical_commissioning_required": False,
            "physical_stop_qualified": False,
            "hand_velocity_rad_s": HAND_SPEED_RAD_S,
            "hand_raw_initial_delta_rad": None,
        }
        self.hands = None

    def bind_hands(self, identities):
        if self.hands is not None or set(identities) != {"left", "right"}:
            raise ValueError("Trial requires one immutable dual-hand identity snapshot")
        for identity in identities.values():
            if not identity.get("serial") or not identity.get("info"):
                raise ValueError("Missing trial hand identity")
            gains, efforts = identity.get("mit_gains", []), identity.get("effort_limits", [])
            if len(gains) != 20 or len(efforts) != 20:
                raise ValueError("Incomplete trial hand operating parameters")
            for gain, effort in zip(gains, efforts, strict=True):
                values = (gain["kp"], gain["kd"], effort)
                if not all(math.isfinite(v) and v > 0 for v in values):
                    raise ValueError("Invalid trial hand operating parameters")
        if identities["left"]["serial"] == identities["right"]["serial"]:
            raise ValueError("Trial hands must be distinct physical devices")
        self.hands = copy.deepcopy(identities)

    def check_hand(self, side, identity):
        if self.hands is None or self.hands.get(side) != identity:
            raise ValueError("Hand identity or operating parameters changed since trial readback: " + side)
