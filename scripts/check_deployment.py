"""Read-only deployment preflight. Never imports SDK writers or connects to devices."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fr3_wuji.paths import CAMERA_CONFIG, CONTROL_IMAGE, TELEOP_PYTHON, TELEOP_ROOT, WUJI_SDK_SITE


def inspect():
    checks = {}
    def check(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": str(detail)}
    for relative in (
        "config/workcell/current.yaml", "config/modes/teleop_control.yaml", "config/cyclonedds.xml",
        "data_collection/contract/dataset_contract.yaml",
        "ros_ws/src/teleop_core/teleop_core/contract.py", "ros_ws/src/teleop_core/teleop_core/safety.py",
        "ros_ws/src/teleop_interfaces/msg/ArmCommand.msg",
        "ros_ws/src/franka_fr3_arm_controllers/src/joint_impedance_controller.cpp",
        "adapters/wuji/config/retarget_manus_wuji_hand_2_left.yaml",
        "adapters/wuji/config/retarget_manus_wuji_hand_2_right.yaml",
    ):
        path = TELEOP_ROOT / relative
        check("teleop:" + relative, path.is_file(), path)
    try:
        import yaml
        cameras = yaml.safe_load(CAMERA_CONFIG.read_text())["camera_bringup"]
        serials = [str(cameras[key]["serial_number"]) for key in ("cam0", "cam1", "cam2")]
        semantics = [cameras[key]["semantic"] for key in ("cam0", "cam1", "cam2")]
        valid = (semantics == ["head", "left_wrist", "right_wrist"] and len(set(serials)) == 3
                 and all(value.strip() and not value.startswith("REPLACE_") for value in serials))
        check("camera_config", valid, CAMERA_CONFIG)
        for key in ("cam0", "cam1", "cam2"):
            preset = TELEOP_ROOT / "ros_ws/src/teleop_camera_bringup/config" / cameras[key]["config_file"]
            check("camera_preset:" + key, preset.is_file(), preset)
    except (OSError, ValueError, KeyError, TypeError) as error:
        check("camera_config", False, error)
    check("teleop_python", TELEOP_PYTHON.is_file(), TELEOP_PYTHON)
    for name in ("wuji_sdk", "wuji_sdk.libs"):
        check(name, (WUJI_SDK_SITE/name).is_dir(), WUJI_SDK_SITE/name)
    try:
        result = subprocess.run(["docker", "image", "inspect", CONTROL_IMAGE], capture_output=True, timeout=10)
        check("control_image", result.returncode == 0, CONTROL_IMAGE)
    except (OSError, subprocess.TimeoutExpired) as error:
        check("control_image", False, error)
    from deployment.profiles import get_profile
    for name in ("slow54", "slow64"):
        profile = get_profile(name)
        try:
            profile.model_adapter().checkpoint_contract(profile.checkpoint)
            check("checkpoint:" + name, True, profile.checkpoint)
        except (OSError, ValueError, KeyError, TypeError) as error:
            check("checkpoint:" + name, False, f"{profile.checkpoint}: {error}")
    overlay = ROOT/".deployment/oneshot-overlay/source-manifest.json"
    try:
        from deployment.hardware import verify_overlay
        verify_overlay()
        check("controller_overlay", True, overlay)
    except Exception as error:
        check("controller_overlay", False, error)
    try:
        from experiments.weight_motion_eval.reference import reference_limits
        limits, names = reference_limits()
        check("joint_limits", len(names) == 54 and limits["lower"].shape == (54,), "54 joint limits read from the local teleop repository")
    except Exception as error:
        check("joint_limits", False, error)
    for name in ("openpi", "openpi_client", "jax", "flax", "numpy", "yaml"):
        spec = importlib.util.find_spec(name)
        location = getattr(spec, "origin", None) or (list(spec.submodule_search_locations or []) if spec else None)
        ok = spec is not None
        if name in {"openpi", "openpi_client"}:
            paths = [spec.origin] if spec and spec.origin else list(spec.submodule_search_locations or []) if spec else []
            ok = bool(paths) and all(Path(p).resolve().is_relative_to(ROOT) for p in paths)
        check("python:" + name, ok, location)
    return checks


if __name__ == "__main__":
    checks = inspect()
    print(json.dumps({"hardware_connected": False, "root": str(ROOT), "checks": checks}, ensure_ascii=False, indent=2))
    sys.exit(0 if all(item["ok"] for item in checks.values()) else 1)
