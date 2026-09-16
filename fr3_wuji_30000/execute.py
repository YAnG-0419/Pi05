"""Compatibility execution entry for slow64; no global module mutations."""
from deployment.profiles import get_profile
from deployment.slow import main as execute

def main(argv=None):
    execute(argv, profile=get_profile("slow64"))

if __name__ == "__main__":
    main()
