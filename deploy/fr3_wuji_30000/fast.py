"""30000v2 entry for the isolated normal-speed deployment pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/30000v2"
DEFAULT_URI = "ws://127.0.0.1:8002"

sys.path.insert(0, str(ROOT))


def with_defaults(argv):
    """Add only the 30000v2-specific defaults; explicit CLI values win."""
    args = list(argv)
    if "--checkpoint" not in args and not any(arg.startswith("--checkpoint=") for arg in args):
        args.extend(("--checkpoint", str(DEFAULT_CHECKPOINT)))
    if "--uri" not in args and not any(arg.startswith("--uri=") for arg in args):
        args.extend(("--uri", DEFAULT_URI))
    if "--arm-motion-limits" not in args and not any(arg.startswith("--arm-motion-limits=") for arg in args):
        args.extend(("--arm-motion-limits", "legacy"))
    return args


def install_fast_profile():
    """Apply the existing 30000 acquisition profile inside this process only."""
    from deploy.fr3_wuji_30000.planning_profile import install
    from experiments.weight_motion_eval import planner

    install()
    # timeline imports read_config by name. Assign it explicitly as well so the
    # profile is effective even when timeline was imported earlier in a test or
    # interactive process.
    from deploy.fr3_wuji_fast import timeline

    timeline.read_config = planner.read_config


def main(argv=None):
    from deploy.fr3_wuji_30000.serve import checkpoint_contract
    from experiments.weight_motion_eval.oneshot import deploy as hardware

    install_fast_profile()
    from deploy.fr3_wuji_fast import deploy as fast_deploy

    # The replacement is process-local. Existing 19999 fast and 30000 slow
    # entry points keep their original defaults and on-disk implementation.
    original_contract = hardware.checkpoint_contract
    hardware.checkpoint_contract = checkpoint_contract
    try:
        fast_deploy.main(with_defaults(sys.argv[1:] if argv is None else argv))
    finally:
        hardware.checkpoint_contract = original_contract


if __name__ == "__main__":
    main()
