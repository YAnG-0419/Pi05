"""Temporary artifacts for launch-preview tests; never usable as a real checkpoint."""
import json
import pytest
from experiments.weight_motion_eval.oneshot.qualification import digest


@pytest.fixture
def launch_artifacts(tmp_path, monkeypatch):
    from deployment.normal import deploy
    overlay = tmp_path / "overlay"
    controller = overlay / "src/franka_fr3_arm_controllers"
    controller.mkdir(parents=True)
    library = overlay / "fake-library-for-test-only.so"
    library.write_bytes(b"test-only")
    (overlay / "source-manifest.json").write_text(json.dumps({
        "build_completed": True, "reference": str(controller), "source_hashes": {},
        "overlay_source_hashes": {}, "built_library": str(library), "built_library_sha256": digest(library),
    }))
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params/manifest.ocdbt").write_bytes(b"test-only")
    (checkpoint / "assets/test").mkdir(parents=True)
    (checkpoint / "assets/test/norm_stats.json").write_text(json.dumps({"norm_stats": {
        key: {"mean": [0.] * 54, "std": [1.] * 54, "q01": [-1.] * 54, "q99": [1.] * 54}
        for key in ("state", "actions")
    }}))
    sdk = tmp_path / "sdk"
    for name in ("wuji_sdk", "wuji_sdk.libs"):
        (sdk/name).mkdir(parents=True)
    monkeypatch.setattr(deploy.hardware, "OVERLAY", overlay)
    monkeypatch.setattr(deploy.hardware, "SDK", sdk)
    monkeypatch.setattr(deploy, "DEFAULT_CHECKPOINT", checkpoint)
    return checkpoint
