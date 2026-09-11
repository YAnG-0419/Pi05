"""Read existing deployment contracts without importing device writers."""

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def deployment_module(name):
    path = str(ROOT / "deploy/fr3_wuji")
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name)


def reference_limits():
    module = deployment_module("record_inference")
    limits = module.load_limits()
    return limits, list(module.NAMES)
