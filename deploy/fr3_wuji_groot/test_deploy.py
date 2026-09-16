import json

import pytest

from deploy.fr3_wuji_groot import deploy


def test_execute_requires_explicit_model_and_hardware_admission():
    with pytest.raises(SystemExit):
        deploy.parse_args(["--execute"])
    with pytest.raises(SystemExit):
        deploy.parse_args(["--execute", "--expected-model-sha256", "a" * 64])


def test_default_mode_is_hardware_free_preview(tmp_path, monkeypatch):
    output = tmp_path / "preview"
    monkeypatch.setattr(deploy, "safe_output", lambda path: path)
    monkeypatch.setattr(
        deploy,
        "write_runtime",
        lambda args, path, model: ({"controller_sha256": "a" * 64}, {"lower": [], "upper": []}),
    )
    monkeypatch.setattr(
        deploy.hardware,
        "launch_commands",
        lambda args, path, ipc: {"preview": ["never-executed"]},
    )
    monkeypatch.setattr(
        deploy.hardware,
        "preflight_owners",
        lambda **kwargs: pytest.fail("offline preview touched hardware preflight"),
    )

    deploy.main(["--output", str(output)])

    settings = json.loads((output / "groot-config.json").read_text())
    commands = json.loads((output / "commands-preview.json").read_text())
    assert settings["hardware_output"] is False
    assert settings["broker_mode"] == "serial_only"
    assert commands == {"preview": ["never-executed"]}


def test_server_check_returns_before_creating_runtime_or_touching_hardware(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.metadata = {"warmed_up": True, "model_manifest_sha256": "a" * 64}

    monkeypatch.setattr(deploy, "GrootClient", FakeClient)
    monkeypatch.setattr(deploy, "safe_output", lambda path: pytest.fail("server check created runtime output"))
    monkeypatch.setattr(
        deploy.hardware,
        "preflight_owners",
        lambda **kwargs: pytest.fail("server check touched hardware preflight"),
    )

    deploy.main(["--server-check"])
