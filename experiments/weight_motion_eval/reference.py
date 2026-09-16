"""Read existing deployment contracts without importing device writers."""

import hashlib
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


def hand_position_limits(side, reference=Path("/home/user/lpy/gello-retarget")):
    """SDK-only diagnostics need no policy-client or camera dependencies."""
    import xml.etree.ElementTree as ET

    import numpy as np
    import yaml

    from deploy.fr3_wuji.observation import joint_names

    if side not in {"left", "right"}:
        raise ValueError("Explicit hand side required")
    config = reference / f"adapters/wuji/config/retarget_manus_wuji_hand_2_{side}.yaml"
    urdf = (config.parent / yaml.safe_load(config.read_text())["optimizer"]["urdf_path"]).resolve()
    nodes = {node.attrib["name"]: node.find("limit") for node in ET.parse(urdf).getroot().findall("joint")}
    names = joint_names(side + "_hand")
    lower = np.array([float(nodes[name].attrib["lower"]) for name in names])
    upper = np.array([float(nodes[name].attrib["upper"]) for name in names])
    if not np.isfinite([lower, upper]).all() or not np.all(lower < upper):
        raise ValueError("Invalid reference hand limits")
    return {
        "lower": lower,
        "upper": upper,
        "names": names,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (config, urdf)},
    }
