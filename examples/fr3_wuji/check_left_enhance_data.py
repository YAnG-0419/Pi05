"""Check the local datasets for the 80/20 left-hand enhancement run.

Run before computing normalization statistics or starting training:
    uv run python -m examples.fr3_wuji.check_left_enhance_data
"""

from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import torch
import tyro

from examples.fr3_wuji.convert_dataset import DEFAULT_TASK
from openpi import transforms
from openpi.training import config as training_config
from openpi.training import data_loader

CAMERA_KEYS = (
    "observation.images.cam0",
    "observation.images.cam1",
    "observation.images.cam2",
)


def _check_dataset(repo_id: str, data_config, horizon: int, *, all_episodes: bool):
    root = Path(HF_LEROBOT_HOME) / repo_id
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(f"Missing local LeRobot dataset: {root}")

    meta = LeRobotDatasetMetadata(repo_id)
    if meta.fps != 30:
        raise ValueError(f"{repo_id}: expected 30 Hz, got {meta.fps}")
    if tuple(meta.shapes.get("observation.state", ())) != (108,):
        raise ValueError(f"{repo_id}: expected 108-dimensional state")
    if tuple(meta.shapes.get("action", ())) != (54,):
        raise ValueError(f"{repo_id}: expected 54-dimensional action")
    if set(meta.camera_keys) != set(CAMERA_KEYS):
        raise ValueError(f"{repo_id}: expected three cam0/cam1/cam2 RGB cameras")
    if set(meta.tasks.values()) != {DEFAULT_TASK}:
        raise ValueError(f"{repo_id}: tasks must use the full deployment instruction exactly")
    if meta.total_episodes <= 0 or meta.total_frames <= 0:
        raise ValueError(f"{repo_id}: dataset is empty")
    for index, episode in meta.episodes.items():
        if episode["length"] <= 0:
            raise ValueError(f"{repo_id}: episode {index} is empty")
        for key in CAMERA_KEYS:
            if meta.features[key]["dtype"] == "video" and not (root / meta.get_video_file_path(index, key)).is_file():
                raise FileNotFoundError(f"{repo_id}: missing video for episode {index}, {key}")

    dataset = LeRobotDataset(
        repo_id,
        video_backend=data_config.video_backend,
        delta_timestamps={"action": [step / meta.fps for step in range(horizon)]},
    )
    if len(dataset) != meta.total_frames:
        raise ValueError(f"{repo_id}: frame count does not match metadata")
    episode_indices = list(meta.episodes)
    selected = (
        episode_indices
        if all_episodes
        else [episode_indices[i] for i in sorted({0, len(episode_indices) // 2, len(episode_indices) - 1})]
    )
    starts = {}
    offset = 0
    for index, episode in meta.episodes.items():
        starts[index] = offset
        offset += episode["length"]

    transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    for episode_index in selected:
        start = starts[episode_index]
        length = meta.episodes[episode_index]["length"]
        for row in sorted({0, length // 2, length - 1}):
            sample = dataset[start + row]
            if int(sample["episode_index"]) != episode_index:
                raise ValueError(f"{repo_id}: sample episode mismatch at row {row}")
            for key in CAMERA_KEYS:
                image = sample[key]
                if image.ndim != 3 or image.shape[0] != 3 or not torch.isfinite(image).all():
                    raise ValueError(f"{repo_id}: invalid RGB image at episode {episode_index}, row {row}, {key}")
            sample["prompt"] = meta.tasks[int(sample["task_index"])]
            converted = transform(sample)
            if converted["state"].shape != (54,) or converted["actions"].shape != (horizon, 54):
                raise ValueError(f"{repo_id}: wrong transformed vector shape at episode {episode_index}, row {row}")
            if not np.isfinite(converted["state"]).all() or not np.isfinite(converted["actions"]).all():
                raise ValueError(f"{repo_id}: non-finite vector at episode {episode_index}, row {row}")
            if converted["prompt"] != DEFAULT_TASK:
                raise ValueError(f"{repo_id}: prompt mismatch at episode {episode_index}, row {row}")

    print(
        f"PASS {repo_id}: {meta.total_episodes} episodes, {meta.total_frames} frames; checked {len(selected)} episodes"
    )
    return dataset


def main(config_name: str = "pi05_fr3_wuji_left_enhance", draws: int = 20_000):
    if draws <= 0:
        raise ValueError("draws must be positive")
    config = training_config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    specs = data_config.lerobot_datasets
    if len(specs) != 4 or not np.allclose([spec.weight for spec in specs], [8 / 15, 4 / 15, 2 / 15, 1 / 15]):
        raise ValueError("Expected complete/left-only datasets with 80/20 weights")
    datasets = [
        _check_dataset(spec.repo_id, data_config, config.model.action_horizon, all_episodes=(i >= 2))
        for i, spec in enumerate(specs)
    ]
    if sum(dataset.meta.total_episodes for dataset in datasets[2:]) != 29:
        raise ValueError("Expected 29 left-only clips across A and B")
    mixture = data_loader.WeightedConcatDataset(datasets, [spec.weight for spec in specs])
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(mixture.sample_weights, dtype=torch.double),
        num_samples=draws,
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    left_rate = sum(index >= sum(len(dataset) for dataset in datasets[:2]) for index in sampler) / draws
    print(f"Left-only sampling: {left_rate:.2%} over {draws} seeded draws (expected 20%)")
    if abs(left_rate - 0.2) > 0.02:
        raise ValueError("Observed left-only sampling rate differs from 20% by more than 2 points")
    print("PASS: data and sampling checks completed. Review first/last clip frames visually before training.")


if __name__ == "__main__":
    tyro.cli(main)
