"""Compatibility entry for the slow54 profile. Shared devices live in deployment.hardware."""
from deployment.slow import main as main
from deployment.slow import parse_args as parse_args
from deployment.hardware import ROOT as ROOT
from deployment.hardware import REFERENCE as REFERENCE
from deployment.hardware import SDK as SDK
from deployment.hardware import OVERLAY as OVERLAY
from deployment.hardware import verify_overlay as verify_overlay
from deployment.hardware import write_runtime as write_runtime
from deployment.hardware import docker_command as docker_command
from deployment.hardware import launch_commands as launch_commands
from deployment.hardware import preflight_owners as preflight_owners
from deployment.hardware import Children as Children

if __name__ == "__main__":
    main()
