"""Compatibility module; implementation moved to deployment.normal.deploy."""
from deployment.normal.deploy import *  # noqa: F403

if __name__ == "__main__":
    main()  # noqa: F405 -- compatibility re-export
