"""Optional one-shot inference through existing read-only observation adapters."""

import json

from .reference import deployment_module


def capture_once(output, *, uri, start_cameras=False, hand_source="ros"):
    # ROS arm feedback avoids compiling or starting a second direct FCI reader.
    recorder_module = deployment_module("record_inference")
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    class OneShot:
        def __init__(self, recorder):
            self.recorder = recorder

        @property
        def completed(self):
            return self.recorder.completed

        def __call__(self, store, obs, metadata, directory):
            if self.completed == 0:
                self.recorder(store, obs, metadata, directory)

    with connect(uri, compression=None, max_size=None, open_timeout=10, close_timeout=2, proxy=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=2))
        expected = {
            "config": "pi05_fr3_wuji",
            "action_dim": 54,
            "action_horizon": 50,
            "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
            "hardware_output": False,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Unexpected model contract")
        recorder = recorder_module.Recorder(ws, recorder_module.load_limits(), timeout=2, max_input_age=0.07)
        consumer = OneShot(recorder)
        args = [
            "--arm-source",
            "ros",
            "--hand-source",
            hand_source,
            "--duration",
            "3",
            "--poll-interval",
            "0.005",
            "--output",
            str(output),
        ]
        if start_cameras:
            args += ["--start-cameras", "--camera-profile", "rgb"]
        error = None
        try:
            recorder_module.observe.main(args, consumer=consumer)
            if consumer.completed != 1:
                raise RuntimeError("No inference completed within the capture window")
            run = recorder.runs[0]
            if run["input_expired_at_return"] or run["return_state_error"]:
                raise RuntimeError("Captured response or feedback was stale; the record is diagnostic only")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if output.exists():
                report = {
                    "mode": "capture_once",
                    "hardware_output": False,
                    "model": metadata,
                    "completed": consumer.completed,
                    "error": error,
                    "status": "captured" if error is None else "failed",
                    "note": "Fresh at capture only; this file is not an authorization for later motion.",
                }
                (output / "capture-report.json").write_text(json.dumps(report, indent=2) + "\n")
