# ruff: noqa: SLF001 -- RTC math is intentionally tested at the small internal helper boundary.

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_rtc_zero_weights_preserve_velocity_exactly():
    x_t = jnp.array([[[0.25, -0.5], [0.75, 1.25]]], dtype=jnp.float32)

    def velocity_fn(actions):
        return 0.25 * actions + 0.125

    baseline_velocity = velocity_fn(x_t)
    guided_velocity = _pi0._rtc_guided_velocity(
        velocity_fn,
        x_t,
        0.5,
        jnp.zeros_like(x_t),
        jnp.zeros((1, 2, 1), dtype=x_t.dtype),
        5.0,
    )

    assert jnp.array_equal(guided_velocity, baseline_velocity)


def test_rtc_guidance_step_reduces_prefix_clean_estimate_error():
    x_t = jnp.array([[[1.0, -1.0], [2.0, -2.0]]], dtype=jnp.float32)
    rtc_actions = jnp.zeros_like(x_t)
    rtc_weights = jnp.array([[[1.0], [0.0]]], dtype=x_t.dtype)

    def velocity_fn(actions):
        return jnp.zeros_like(actions)

    guided_velocity = _pi0._rtc_guided_velocity(
        velocity_fn,
        x_t,
        0.5,
        rtc_actions,
        rtc_weights,
        2.0,
    )
    x_next = x_t - 0.1 * guided_velocity

    error_before = jnp.linalg.norm((rtc_actions - x_t) * rtc_weights)
    error_after = jnp.linalg.norm((rtc_actions - x_next) * rtc_weights)
    assert error_after < error_before


def test_rtc_guidance_is_finite_at_time_endpoints():
    x_t = jnp.array([[[1.0, -1.0]]], dtype=jnp.float32)
    rtc_actions = jnp.zeros_like(x_t)
    rtc_weights = jnp.ones((1, 1, 1), dtype=x_t.dtype)

    for time in (0.0, 1.0):
        guided_velocity = _pi0._rtc_guided_velocity(
            jnp.zeros_like,
            x_t,
            time,
            rtc_actions,
            rtc_weights,
            5.0,
        )
        assert jnp.all(jnp.isfinite(guided_velocity))
