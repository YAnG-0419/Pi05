"""Portable, offline-only GR00T base-model cache validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

MODEL_IDS = ("nvidia/GR00T-N1.7-3B", "nvidia/Cosmos-Reason2-2B")
_REVISION = re.compile(r"[0-9a-f]{40}")


def configure_model_cache(cache: Path) -> dict[str, dict[str, str]]:
    """Validate a moved HF cache and force all later model loading offline.

    ``models.json`` records absolute snapshot paths from the machine that
    downloaded the models.  Those paths are provenance only: deployment
    reconstructs each snapshot from its pinned revision under the mounted
    cache, so the same cache can be moved into a container or another host.
    """

    cache = Path(cache).expanduser().resolve()
    manifest_path = cache / "models.json"
    report = json.loads(manifest_path.read_text())
    if report.get("complete") is not True or not isinstance(report.get("models"), dict):
        raise ValueError("GR00T base-model cache manifest is incomplete")
    hub = (cache / "hub").resolve()
    resolved = {}
    for model_id in MODEL_IDS:
        entry = report["models"].get(model_id)
        if not isinstance(entry, dict):
            raise ValueError(f"GR00T base-model cache is missing {model_id}")
        revision = entry.get("revision")
        if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
            raise ValueError(f"Invalid pinned revision for {model_id}")
        repository = hub / ("models--" + model_id.replace("/", "--"))
        reference = repository / "refs/main"
        if not reference.is_file() or reference.read_text().strip() != revision:
            raise ValueError(f"Cached main revision does not match the recorded revision for {model_id}")
        snapshot = (repository / "snapshots" / revision).resolve()
        if not snapshot.is_relative_to(hub) or not (snapshot / "config.json").is_file():
            raise ValueError(f"Pinned snapshot is unavailable for {model_id}")
        resolved[model_id] = {"revision": revision, "snapshot": str(snapshot)}

    os.environ.update(
        {
            "HF_HOME": str(cache),
            "HF_HUB_CACHE": str(hub),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return resolved
