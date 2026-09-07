import numpy as np
import pytest

import openpi.models.model as _model
from openpi.training import weight_loaders


def test_partial_checkpoint_loader_keeps_mismatched_projection(monkeypatch):
    reference = {
        "action_in_proj": {"kernel": np.ones((54, 4), dtype=np.float32)},
        "backbone": {"kernel": np.ones((4, 4), dtype=np.float32)},
    }
    loaded = {
        "action_in_proj": {"kernel": np.zeros((32, 4), dtype=np.float32)},
        "backbone": {"kernel": np.zeros((4, 4), dtype=np.float32)},
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader("checkpoint").load(reference)

    np.testing.assert_array_equal(result["action_in_proj"]["kernel"], reference["action_in_proj"]["kernel"])
    np.testing.assert_array_equal(result["backbone"]["kernel"], loaded["backbone"]["kernel"])


def test_partial_checkpoint_loader_rejects_other_shape_mismatch(monkeypatch):
    reference = {"backbone": {"kernel": np.ones((4, 4), dtype=np.float32)}}
    loaded = {"backbone": {"kernel": np.zeros((3, 4), dtype=np.float32)}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(_model, "restore_params", lambda *args, **kwargs: loaded)

    with pytest.raises(ValueError, match="Shape mismatch"):
        weight_loaders.PartialCheckpointWeightLoader("checkpoint").load(reference)
