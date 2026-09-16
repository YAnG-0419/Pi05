"""Compatibility model-service entry for the slow64 profile."""
from deployment.models.joint64 import *  # noqa: F403
from deployment.server import main as serve

def main():
    serve(default_profile="slow64")

if __name__ == "__main__":
    main()
