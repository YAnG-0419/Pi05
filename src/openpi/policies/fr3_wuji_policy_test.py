import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import fr3_wuji_policy


def test_inputs_reorder_state_actions_and_images():
    state = np.arange(108, dtype=np.float32)
    actions = np.arange(2 * 54, dtype=np.float32).reshape(2, 54)
    transform = fr3_wuji_policy.Fr3WujiInputs(model_type=_model.ModelType.PI05)

    result = transform(
        {
            "observation/state": state,
            "observation/image": np.ones((3, 4, 5), dtype=np.float32),
            "observation/left_wrist_image": np.zeros((4, 5, 3), dtype=np.uint8),
            "observation/right_wrist_image": np.zeros((4, 5, 3), dtype=np.uint8),
            "actions": actions,
            "prompt": "test",
        }
    )

    expected_state = np.concatenate([state[0:7], state[28:48], state[14:21], state[68:88]])
    expected_actions = np.concatenate(
        [actions[..., 0:7], actions[..., 14:34], actions[..., 7:14], actions[..., 34:54]], axis=-1
    )
    np.testing.assert_array_equal(result["state"], expected_state)
    np.testing.assert_array_equal(result["actions"], expected_actions)
    assert result["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert all(result["image_mask"].values())
    assert result["prompt"] == "test"


def test_inputs_accept_54_dim_state():
    state = np.arange(54, dtype=np.float32)
    result = fr3_wuji_policy.Fr3WujiInputs(model_type=_model.ModelType.PI05)(
        {
            **fr3_wuji_policy.make_fr3_wuji_example(),
            "observation/state": state,
        }
    )
    np.testing.assert_array_equal(result["state"], state)


@pytest.mark.parametrize("state_dim", [53, 109])
def test_inputs_reject_unexpected_state_dimension(state_dim):
    example = fr3_wuji_policy.make_fr3_wuji_example()
    example["observation/state"] = np.zeros(state_dim)
    with pytest.raises(ValueError, match="state dimension"):
        fr3_wuji_policy.Fr3WujiInputs(model_type=_model.ModelType.PI05)(example)


def test_outputs_keep_54_dimensions():
    actions = np.zeros((50, 60))
    result = fr3_wuji_policy.Fr3WujiOutputs()({"actions": actions})
    assert result["actions"].shape == (50, 54)
