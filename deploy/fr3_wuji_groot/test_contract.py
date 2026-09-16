import json

import numpy as np
import pytest

from deploy.fr3_wuji_groot import contract
from deploy.fr3_wuji_groot.model_server import checkpoint_contract


def test_joint_order_round_trip_and_group_boundaries():
    physical = np.arange(54, dtype=np.float32)
    groot = contract.physical_to_groot(physical)
    np.testing.assert_array_equal(groot[:7], physical[:7])
    np.testing.assert_array_equal(groot[7:14], physical[27:34])
    np.testing.assert_array_equal(groot[14:34], physical[7:27])
    np.testing.assert_array_equal(groot[34:], physical[34:])
    physical_actions = contract.groot_to_physical(np.tile(groot, (contract.ACTION_HORIZON, 1)))
    np.testing.assert_array_equal(physical_actions, np.tile(physical, (contract.ACTION_HORIZON, 1)))


@pytest.mark.parametrize(
    "value",
    [np.zeros(53), np.zeros(55), np.full(54, np.nan), np.zeros((1, 54))],
)
def test_invalid_state_is_rejected(value):
    with pytest.raises(ValueError, match="54 finite"):
        contract.finite_vector(value)


def make_checkpoint(path):
    statistics = {"new_embodiment": {"state": {"mean": [0.0] * 54}}}
    (path / "statistics.json").write_text(json.dumps(statistics))
    digest = contract.semantic_json_sha256(path / "statistics.json")
    (path / "inference_recipe.json").write_text(
        json.dumps(
            {
                "statistics_sha256": digest,
                "action_head_dtype": "float32",
                "padding_matching": True,
                "full_task_instruction": "pick tomatoes",
                "num_inference_timesteps": 4,
            }
        )
    )
    (path / "reload_verification.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "statistics_sha256": digest,
                "output_shape": [16, 54],
                "max_difference": 0.0,
                "backbone_dtype": "bfloat16",
            }
        )
    )
    for name in ("config.json", "embodiment_id.json", "processor_config.json"):
        (path / name).write_text("{}")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"test-only-weights")
    return digest


def test_checkpoint_contract_binds_statistics_and_all_weight_files(tmp_path):
    statistics = make_checkpoint(tmp_path)
    metadata, manifest = checkpoint_contract(tmp_path)
    assert metadata["statistics_sha256"] == statistics
    assert len(metadata["model_manifest_sha256"]) == 64
    assert {entry["path"] for entry in manifest} >= {
        "config.json",
        "statistics.json",
        "model-00001-of-00001.safetensors",
    }
    metadata["warmed_up"] = True
    contract.validate_metadata(metadata, expected_model_sha256=metadata["model_manifest_sha256"])


def test_checkpoint_contract_rejects_unverified_reload(tmp_path):
    make_checkpoint(tmp_path)
    verification = json.loads((tmp_path / "reload_verification.json").read_text())
    verification["max_difference"] = 0.001
    (tmp_path / "reload_verification.json").write_text(json.dumps(verification))
    with pytest.raises(ValueError, match="exact reload"):
        checkpoint_contract(tmp_path)
