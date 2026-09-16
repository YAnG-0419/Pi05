from http.server import HTTPServer
import threading

import numpy as np
import pytest

from deploy.fr3_wuji_groot.client import GrootClient
from deploy.fr3_wuji_groot.contract import ACTION_ORDER
from deploy.fr3_wuji_groot.contract import PROTOCOL
from deploy.fr3_wuji_groot.model_server import GrootInference
from deploy.fr3_wuji_groot.model_server import make_handler


class Policy:
    def __init__(self):
        self.calls = []

    def predict(self, images, joints, prompt):
        self.calls.append((images, joints, prompt))
        groups = [
            np.full((16, 7), 1, dtype=np.float32),
            np.full((16, 7), 2, dtype=np.float32),
            np.full((16, 20), 3, dtype=np.float32),
            np.full((16, 20), 4, dtype=np.float32),
        ]
        return np.concatenate(groups, axis=-1)


def metadata():
    return {
        "protocol": PROTOCOL,
        "backend": "groot_n1d7",
        "model_manifest_sha256": "a" * 64,
        "statistics_sha256": "b" * 64,
        "action_shape": [16, 54],
        "action_hz": 30,
        "action_order": ACTION_ORDER,
        "joint_units": "radians",
        "action_semantics": "absolute_joint_targets",
        "reload_verified": True,
        "warmed_up": True,
        "hardware_output": False,
        "instruction": "pick tomatoes",
    }


@pytest.fixture
def server():
    policy = Policy()
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(GrootInference(policy), metadata()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", policy
    finally:
        httpd.shutdown()
        thread.join(2)
        httpd.server_close()


def observation():
    state = np.concatenate(
        [
            np.full(7, 10, dtype=np.float32),
            np.full(20, 20, dtype=np.float32),
            np.full(7, 30, dtype=np.float32),
            np.full(20, 40, dtype=np.float32),
        ]
    )
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    return {
        "observation/state": state,
        "observation/image": image,
        "observation/left_wrist_image": image,
        "observation/right_wrist_image": image,
        "prompt": "this unrelated Pi05 prompt must be ignored",
    }


def test_client_server_preserves_physical_contract(server):
    address, policy = server
    client = GrootClient(address, expected_model_sha256="a" * 64)
    result = client.infer(observation())
    actions = result["actions"]
    assert actions.shape == (16, 54)
    np.testing.assert_array_equal(actions[:, :7], 1)
    np.testing.assert_array_equal(actions[:, 7:27], 3)
    np.testing.assert_array_equal(actions[:, 27:34], 2)
    np.testing.assert_array_equal(actions[:, 34:], 4)
    _, joints, prompt = policy.calls[0]
    np.testing.assert_array_equal(joints["left_arm"], 10)
    np.testing.assert_array_equal(joints["left_hand"], 20)
    np.testing.assert_array_equal(joints["right_arm"], 30)
    np.testing.assert_array_equal(joints["right_hand"], 40)
    assert prompt == "pick tomatoes"


def test_wrong_model_is_rejected_during_client_construction(server):
    address, _ = server
    with pytest.raises(ValueError, match="explicitly admitted checkpoint"):
        GrootClient(address, expected_model_sha256="c" * 64)
