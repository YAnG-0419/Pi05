"""Portability checks without Docker or robot connections."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from dp_deploy.build import prepare_build
from dp_deploy.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_old_overlay_rejected_until_fresh_build(tmp_path):
    output = tmp_path / "overlay"
    output.mkdir()
    reference = tmp_path / "reference"
    (output / "source-manifest.json").write_text(
        json.dumps(
            {
                "reference": "/old-deployment/reference",
                "built_library": "/old-deployment/overlay/library.so",
            }
        )
    )
    (output / "old.so").write_bytes(b"preserved")
    with pytest.raises(ValueError, match="Relocated"):
        prepare_build(output, reference)
    prepare_build(output, reference, fresh=True)
    assert not output.exists()
    backups = list(tmp_path.glob("overlay-backup-*"))
    assert len(backups) == 1 and (backups[0] / "old.so").read_bytes() == b"preserved"


def test_matching_overlay_preserved(tmp_path):
    output = tmp_path / "overlay"
    output.mkdir()
    reference = tmp_path / "reference"
    library = (
        output
        / "install/franka_fr3_arm_controllers/lib/libfranka_fr3_arm_controllers.so"
    )
    (output / "source-manifest.json").write_text(
        json.dumps(
            {
                "reference": str(reference),
                "built_library": str(library),
            }
        )
    )
    prepare_build(output, reference)
    assert output.exists() and not list(tmp_path.glob("overlay-backup-*"))


def test_environment_paths_override_deployment_defaults(monkeypatch, tmp_path):
    for key, env in [
        ("reference", "DP_REFERENCE"),
        ("checkpoint", "DP_CHECKPOINT"),
        ("sdk_site", "DP_SDK_SITE"),
    ]:
        monkeypatch.setenv(env, str(tmp_path / key))
    monkeypatch.setenv("DP_DOCKER_IMAGE", "existing-robot-image:latest")
    c = load_config()
    assert c["reference"] == str(tmp_path / "reference")
    assert c["checkpoint"] == str(tmp_path / "checkpoint")
    assert c["sdk_site"] == str(tmp_path / "sdk_site")
    assert c["docker_image"] == "existing-robot-image:latest"


def test_relocated_launcher_uses_local_venv_from_any_cwd(tmp_path):
    root = tmp_path / "deployment with spaces"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    shutil.copy2(ROOT / "run.sh", root / "run.sh")
    shutil.copy2(ROOT / "scripts/python_env.sh", root / "scripts/python_env.sh")
    python = root / ".venv/bin/python"
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
    python.chmod(0o755)
    env = os.environ.copy()
    env.pop("DP_PYTHON", None)
    result = subprocess.run(
        ["bash", str(root / "run.sh"), "serve", "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [str(root), "-m", "dp_deploy.serve", "--help"]


def test_missing_python_has_actionable_error(tmp_path):
    env = os.environ.copy()
    env.pop("DP_PYTHON", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'DP_ROOT="$1"; source "$2"',
            "check",
            str(tmp_path),
            str(ROOT / "scripts/python_env.sh"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "setup_env.sh" in result.stderr
