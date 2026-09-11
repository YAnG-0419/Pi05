"""FR3/Wuji client: replay and live shadow, with hardware output unavailable by default."""

import argparse
import json
from pathlib import Path

from executor.config import load_config
from executor.replay import run_replay

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "deploy/fr3_wuji/executor.example.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("replay", help="Inspect existing observations/actions on a virtual clock")
    replay.add_argument("--source", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--inject", choices=("late_response", "feedback_loss", "pause_before_response"))
    live = commands.add_parser("live-shadow", help="Use read-only live producers and a separate inference process")
    live.add_argument("--output", type=Path, required=True)
    live.add_argument(
        "--single-shadow-step", action="store_true", help="Request one in-memory trial; never writes hardware"
    )
    args, observation_args = parser.parse_known_args()
    config = load_config(args.config)
    if args.command == "replay":
        if observation_args:
            parser.error(f"Unexpected replay arguments: {observation_args}")
        summary = run_replay(args.source, args.output, config, inject=args.inject)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        from executor.live import run_live

        run_live(args.output, config, observation_args, single_step=args.single_shadow_step)


if __name__ == "__main__":
    main()
