"""DP3 entry point using the existing qualified slow hardware bridge."""

from deployment.profiles import DP3_PROFILE
from deployment.slow import main as slow_main


def main(argv=None):
    slow_main(argv, profile=DP3_PROFILE)


if __name__ == "__main__":
    main()
