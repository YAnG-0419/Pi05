# ruff: noqa: SLF001 -- Verify that the public RTC inverse matches the internal dataset reorder.

import numpy as np
import pytest

from openpi import transforms
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


def test_rtc_hardware_actions_round_trip_through_dataset_order():
    actions = np.concatenate(
        [
            np.full((3, 7), 1.0),
            np.full((3, 20), 2.0),
            np.full((3, 7), 3.0),
            np.full((3, 20), 4.0),
        ],
        axis=-1,
    )

    dataset_order = fr3_wuji_policy.rtc_actions_to_dataset_order(actions)

    np.testing.assert_array_equal(dataset_order[:, :7], 1.0)
    np.testing.assert_array_equal(dataset_order[:, 7:14], 3.0)
    np.testing.assert_array_equal(dataset_order[:, 14:34], 2.0)
    np.testing.assert_array_equal(dataset_order[:, 34:], 4.0)
    np.testing.assert_array_equal(fr3_wuji_policy._reorder_actions(dataset_order), actions)


def test_rtc_prefix_reanchors_arm_deltas_to_current_observation():
    current_state = np.arange(54, dtype=np.float32) / 10
    physical_actions = np.tile(current_state + 0.5, (3, 1))
    raw_actions = fr3_wuji_policy.rtc_actions_to_dataset_order(physical_actions)
    example = fr3_wuji_policy.make_fr3_wuji_example()
    example.update({"observation/state": current_state, "actions": raw_actions})
    model_inputs = fr3_wuji_policy.Fr3WujiInputs(model_type=_model.ModelType.PI05)(example)
    delta = transforms.DeltaActions(mask=transforms.make_bool_mask(7, -20, 7, -20))(model_inputs)

    np.testing.assert_allclose(delta["actions"][:, :7], 0.5)
    np.testing.assert_allclose(delta["actions"][:, 27:34], 0.5)
    np.testing.assert_array_equal(delta["actions"][:, 7:27], physical_actions[:, 7:27])
    np.testing.assert_array_equal(delta["actions"][:, 34:54], physical_actions[:, 34:54])
