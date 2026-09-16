"""Checks one episode24 sample with the pi05_fr3_wuji online transforms."""

import json
import pathlib

import numpy as np

from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main() -> None:
    config = _config.get_config("pi05_fr3_wuji")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    raw = dataset[0]

    assert len(dataset) == 2377, f"Expected 2377 frames, got {len(dataset)}"
    info_path = pathlib.Path(data_config.repo_id) / "meta/info.json"
    features = json.loads(info_path.read_text())["features"]
    assert "observation.depths.cam0" in features, "Expected the source depth column to exist"

    repacked = _transforms.compose(data_config.repack_transforms.inputs)(raw)
    reordered = data_config.data_transforms.inputs[0](repacked)
    transformed = _transforms.compose(data_config.data_transforms.inputs)(repacked)

    raw_action = np.asarray(raw["action"])
    np.testing.assert_array_equal(reordered["actions"][..., 0:7], raw_action[..., 0:7])
    np.testing.assert_array_equal(reordered["actions"][..., 7:27], raw_action[..., 14:34])

    assert reordered["state"].shape[-1] == 54
    assert transformed["actions"].shape == (config.model.action_horizon, 54)
    assert "observation.depths.cam0" not in transformed
    for name, image in reordered["image"].items():
        assert image.ndim == 3
        assert image.shape[-1] == 3
        assert image.dtype == np.uint8
        print(f"{name}: shape={image.shape}, dtype={image.dtype}")

    print(f"frames={len(dataset)}, state={reordered['state'].shape}, actions={transformed['actions'].shape}")
    print("FR3/Wuji stage 1 data check passed.")


if __name__ == "__main__":
    main()
