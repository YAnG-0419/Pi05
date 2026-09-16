from pathlib import Path

import yaml

ROOT = Path(__file__).parent


def test_compose_keeps_model_container_away_from_robot_devices():
    compose = yaml.safe_load((ROOT / "compose.groot.yaml").read_text())
    service = compose["services"]["groot-policy"]
    assert "network_mode" not in service
    assert service["networks"] == ["inference-only"]
    assert compose["networks"]["inference-only"]["internal"] is True
    assert service["ports"] == ["127.0.0.1:${GROOT_PORT:-8766}:8766"]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service.get("privileged") is None
    assert all(str(volume).endswith(":ro") for volume in service["volumes"])
    assert not any("/dev" in str(volume) for volume in service["volumes"])


def test_inference_image_does_not_copy_model_artifacts():
    dockerfile = (ROOT / "Dockerfile.inference").read_text()
    assert "COPY --from=groot_src" in dockerfile
    assert "COPY deploy/fr3_wuji_groot" in dockerfile
    assert "checkpoint" not in dockerfile.lower()
    assert "outputs/" not in dockerfile
