from collections.abc import Callable, Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import rtc as _rtc
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        rtc_action_repack: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            rtc_action_repack: Converts deployed absolute actions into the raw
                action layout consumed by the input transform. Policies without
                this adapter reject RTC requests.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_action_repack = rtc_action_repack

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc_context: _rtc.RTCContext | dict | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        context = None
        if rtc_context is not None:
            if self._is_pytorch_model:
                raise ValueError("RTC soft prefix guidance is only implemented for JAX policies")
            if self._rtc_action_repack is None:
                raise ValueError("This policy has no RTC action-layout adapter")
            context = _rtc.validate_context(rtc_context)
            inputs["actions"] = self._rtc_action_repack(context.previous_actions)
        inputs = self._input_transform(inputs)

        rtc_target = None
        if context is not None:
            transformed_prefix = np.asarray(inputs.pop("actions"))
            expected_dim = self._model.action_dim
            expected_horizon = self._model.action_horizon
            if transformed_prefix.ndim != 2 or transformed_prefix.shape != (
                len(context.previous_actions),
                expected_dim,
            ):
                raise ValueError(
                    "RTC input transforms must preserve the prefix length and produce "
                    f"{expected_dim} action dimensions, got {transformed_prefix.shape}"
                )
            if not np.isfinite(transformed_prefix).all():
                raise ValueError("RTC input transforms produced non-finite actions")
            if expected_horizon != _rtc.ACTION_HORIZON:
                raise ValueError("RTC protocol and model action horizons do not match")
            rtc_target = np.zeros((expected_horizon, expected_dim), dtype=transformed_prefix.dtype)
            prefix_length = min(len(transformed_prefix), expected_horizon)
            rtc_target[:prefix_length] = transformed_prefix[:prefix_length]

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if context is not None:
            weights = _rtc.prefix_weights(
                context.inference_delay_steps,
                context.prefix_attention_horizon,
                context.prefix_attention_schedule,
            )
            sample_kwargs.update(
                rtc_actions=jnp.asarray(rtc_target)[None, ...],
                rtc_weights=jnp.asarray(weights, dtype=jnp.float32)[None, :, None],
                rtc_max_guidance_weight=jnp.asarray(context.max_guidance_weight, dtype=jnp.float32),
            )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if context is not None:
            outputs[_rtc.RTC_RECEIPT_KEY] = _rtc.make_receipt(context)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
