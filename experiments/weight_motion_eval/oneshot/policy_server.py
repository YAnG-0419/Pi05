"""Compatibility service entry for the 54D model adapter."""
from deployment.models.joint54 import checkpoint_contract as checkpoint_contract
from deployment.server import main

if __name__ == "__main__":
    main(default_profile="slow54")
