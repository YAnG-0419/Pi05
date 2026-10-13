import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import rtc
import pytest

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import fr3_wuji_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_rtc_policy_transforms_prefix_to_fixed_model_space_and_returns_receipt(monkeypatch):
    class FakeModel:
        action_dim = 54
        action_horizon = 50

        def __init__(self):
            self.sample_kwargs = None

        def sample_actions(self, rng, observation, **kwargs):
            del rng
            self.sample_kwargs = kwargs
            return jnp.zeros((observation.state.shape[0], self.action_horizon, self.action_dim), dtype=jnp.float32)

    # Keep this a transform/Policy contract test; model-side JIT/VJP behavior is
    # covered in pi0_test.py and does not require a real NNX model here.
    monkeypatch.setattr(_policy.nnx_utils, "module_jit", lambda method: method)
    model = FakeModel()
    action_mean = np.arange(54, dtype=np.float32) / 100
    action_std = np.linspace(1.0, 2.0, 54, dtype=np.float32)
    policy = _policy.Policy(
        model,
        transforms=[
            fr3_wuji_policy.Fr3WujiInputs(model_type=_model.ModelType.PI05),
            transforms.DeltaActions(transforms.make_bool_mask(7, -20, 7, -20)),
            transforms.Normalize(
                {"actions": _normalize.NormStats(mean=action_mean, std=action_std)},
            ),
        ],
        rtc_action_repack=fr3_wuji_policy.rtc_actions_to_dataset_order,
    )

    state = np.arange(54, dtype=np.float32) / 10
    previous_actions = np.stack([state + offset for offset in (0.5, 1.0, 1.5)])
    context = rtc.validate_context(
        {
            "protocol_version": rtc.PROTOCOL_VERSION,
            "token": "policy-transform-test",
            "source_number": 1,
            "source_sha256": "a" * 64,
            "previous_actions": previous_actions,
            "inference_delay_steps": 1,
            "prefix_attention_horizon": 3,
            "prefix_attention_schedule": "linear",
            "max_guidance_weight": 7.0,
        }
    )
    observation = {
        "observation/state": state,
        "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    result = policy.infer(observation, rtc_context=context)

    model_space = previous_actions.copy()
    model_space[:, :7] -= state[:7]
    model_space[:, 27:34] -= state[27:34]
    expected_prefix = (model_space - action_mean) / (action_std + 1e-6)
    assert model.sample_kwargs is not None
    rtc_actions = np.asarray(model.sample_kwargs["rtc_actions"])
    rtc_weights = np.asarray(model.sample_kwargs["rtc_weights"])
    assert rtc_actions.shape == (1, 50, 54)
    assert rtc_weights.shape == (1, 50, 1)
    np.testing.assert_allclose(rtc_actions[0, :3], expected_prefix, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(rtc_actions[0, 3:], 0.0)
    np.testing.assert_array_equal(rtc_weights[0, :, 0], rtc.prefix_weights(1, 3, "linear"))
    assert np.asarray(model.sample_kwargs["rtc_max_guidance_weight"]).shape == ()
    assert float(model.sample_kwargs["rtc_max_guidance_weight"]) == 7.0
    rtc.validate_receipt(result[rtc.RTC_RECEIPT_KEY], context)
