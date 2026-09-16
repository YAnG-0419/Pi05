"""Select a supported execution profile without mutating another implementation."""
import argparse
from deployment.profiles import MODELS, PROFILES, STRATEGIES, compose_profile, get_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--profile', choices=tuple(PROFILES), help='Legacy combined preset')
    choice.add_argument('--strategy', choices=STRATEGIES)
    parser.add_argument('--model', choices=tuple(MODELS))
    parser.add_argument('--planning-limits', choices=('standard', 'extended'))
    args, remaining = parser.parse_known_args(argv)
    if args.profile:
        if args.model or args.planning_limits:
            parser.error('Use --strategy when selecting a model or planning limits separately')
        profile = get_profile(args.profile)
    else:
        limits = 'slow64' if args.planning_limits == 'extended' else 'slow54'
        try:
            profile = compose_profile(args.strategy, args.model or '19999', planning_profile=limits)
        except ValueError as error:
            parser.error(str(error))
    if profile.engine == 'slow':
        from deployment.slow import main as execute
        execute(remaining, profile=profile)
    else:
        from deployment.normal.deploy import main as execute
        execute(remaining, profile=profile)


if __name__ == '__main__':
    main()
