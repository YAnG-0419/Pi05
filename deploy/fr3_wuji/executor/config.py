"""Runtime configuration. This release never opens the unqualified execute path."""

from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Unsupported executor config")
    if config.get("mode") not in {"replay_shadow", "live_shadow", "execute"}:
        raise ValueError("Unknown executor mode")
    if config.get("hardware_output") is not False or config["mode"] == "execute":
        raise ValueError(
            "Real execution is unavailable: final-consumer deadlines, stop behaviour and acquisition "
            "must be qualified first. Changing YAML booleans cannot enable it."
        )
    if config.get("continuous_execution_enabled") is not False:
        raise ValueError("Continuous execution is not implemented")
    expected = {
        "max_age_at_send_ms": 70,
        "max_age_at_response_ms": 200,
        "max_latest_feedback_age_ms": 150,
        "max_skew_ms": 80,
        "clock_jump_ms": 50,
    }
    if any(config.get("observation", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("Observation safety settings must match the validated configuration")
    if config.get("arm", {}).get("initial_delta_rad") != 0.05:
        raise ValueError("Arm acquisition threshold must remain 0.05 rad")
    if config.get("arm", {}).get("max_joint_speed_rad_s") != 0.7:
        raise ValueError("Arm speed reference must remain 0.7 rad/s")
    model = config.get("model", {})
    model_expected = {
        "config": "pi05_fr3_wuji",
        "action_dim": 54,
        "horizon": 50,
        "action_sample_hz": 30,
        "inflight_requests": 1,
    }
    if any(model.get(key) != value for key, value in model_expected.items()):
        raise ValueError("Model contract differs from the supported 54-dimensional policy")
    if config.get("scheduler", {}).get("max_policy_steps_per_trial") != 1:
        raise ValueError("Only a single shadow trial step is supported")
    scheduler_expected = {
        "proposed_tick_hz": 100,
        "max_original_observation_age_at_dispatch_ms": 200,
        "pending_candidate_capacity": 1,
        "acquisition_policy": "reject",
    }
    scheduler = config.get("scheduler", {})
    if any(scheduler.get(key) != value for key, value in scheduler_expected.items()):
        raise ValueError("Scheduler settings differ from the implemented shadow policy")
    if (
        scheduler.get("policy_dispatch_requires_original_observation_fresh") is not True
        or scheduler.get("drop_late_generations") is not True
        or scheduler.get("catch_up_missed_ticks") is not False
    ):
        raise ValueError("Freshness, generation checks and no catch-up must remain enabled")
    return config
