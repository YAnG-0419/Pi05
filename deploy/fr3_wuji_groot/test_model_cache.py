import json
import os

import pytest

from deploy.fr3_wuji_groot.model_cache import MODEL_IDS
from deploy.fr3_wuji_groot.model_cache import configure_model_cache


def make_cache(path):
    models = {}
    for index, model_id in enumerate(MODEL_IDS, 1):
        revision = str(index) * 40
        repository = path / "hub" / ("models--" + model_id.replace("/", "--"))
        (repository / "refs").mkdir(parents=True)
        (repository / "refs/main").write_text(revision)
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        models[model_id] = {
            "revision": revision,
            "snapshot": f"/old/machine/path/that/must/not/be/used/{index}",
        }
    (path / "models.json").write_text(json.dumps({"complete": True, "models": models}))


def test_moved_cache_uses_revision_instead_of_recorded_absolute_path(tmp_path, monkeypatch):
    make_cache(tmp_path)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    result = configure_model_cache(tmp_path)
    assert set(result) == set(MODEL_IDS)
    assert all(str(tmp_path.resolve()) in details["snapshot"] for details in result.values())
    assert result[MODEL_IDS[0]]["snapshot"] != "/old/machine/path/that/must/not/be/used/1"
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_cache_rejects_revision_mismatch(tmp_path):
    make_cache(tmp_path)
    model_id = MODEL_IDS[0]
    repository = tmp_path / "hub" / ("models--" + model_id.replace("/", "--"))
    (repository / "refs/main").write_text("f" * 40)
    with pytest.raises(ValueError, match="does not match"):
        configure_model_cache(tmp_path)
