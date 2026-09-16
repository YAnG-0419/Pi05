"""Process-local 30000 acquisition policy; original planner files stay unchanged."""
import math

ARM_INITIAL_LIMIT_RAD = 1.5
ARM_SMOOTHING_LIMIT_RAD = 0.2


def install():
    from experiments.weight_motion_eval import planner
    if getattr(planner, '_pi05_30000_profile', False):
        return
    original_validate = planner.validate_config
    original_read = planner.read_config

    def validate_config(config):
        if not isinstance(config, dict):
            return original_validate(config)
        value = config.get('arm_raw_initial_delta_rad')
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= ARM_INITIAL_LIMIT_RAD):
            raise ValueError(f'30000 arm_raw_initial_delta_rad must be in (0, {ARM_INITIAL_LIMIT_RAD}] rad')
        # Delegate EVERY other configuration check to the existing validator.
        # Its legacy 0.3 ceiling applies only to this validation copy; the
        # trajectory builder receives the real 1.5 bound and checks raw deltas.
        original_validate(dict(config, arm_raw_initial_delta_rad=min(value, 0.3)))
        return config

    def read_config(path):
        config = original_read(path)
        return validate_config(dict(
            config, arm_raw_initial_delta_rad=ARM_INITIAL_LIMIT_RAD,
            max_smoothing_delta_rad=dict(config['max_smoothing_delta_rad'], arm=ARM_SMOOTHING_LIMIT_RAD),
        ))

    planner.validate_config = validate_config
    planner.read_config = read_config
    planner._pi05_30000_profile = True
