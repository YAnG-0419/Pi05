"""Fast profile adapter for the shared IPC client and explicit arm policy."""

from deploy.fr3_wuji_runtime.ipc import DeviceClient
from deploy.fr3_wuji_slow.core import Feedback
from deploy.fr3_wuji_slow.core import checked_arm_speed


class RemoteDevices(DeviceClient):
    feedback_type = Feedback

    def __init__(
        self,
        path,
        *,
        execute=False,
        finish_policy="hold",
        sock=None,
        arm_speed_enforced=False,
        arm_speed_rad_s=None,
    ):
        super().__init__(
            path, feedback_type=Feedback, speed_validator=checked_arm_speed,
            execute=execute, finish_policy=finish_policy, sock=sock,
            arm_speed_enforced=arm_speed_enforced,
            arm_speed_rad_s=arm_speed_rad_s,
        )
