"""Compute FR3/Wuji vector statistics without decoding every video frame.

Uses the configured repack/state/action/delta transforms and LeRobot's forward
action windows, clamped at each episode boundary. All frames, including the final
partial batch, are included. Boundary/middle samples are checked against LeRobot.
Only the known image-independent FR3/Wuji transforms are supported.

Example:
    uv run examples/fr3_wuji/compute_norm_stats.py --config-name pi05_fr3_wuji_125ep
"""

import json

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

from openpi import transforms
from openpi.policies import fr3_wuji_policy
from openpi.shared import normalize
from openpi.training import config as training_config
from openpi.training import data_loader


def transform_vectors(state, actions, rows, horizon, transform):
    indices = np.minimum(rows[:, None] + np.arange(horizon), len(state) - 1)
    # Images cannot affect state/actions for the explicitly allowed transforms below.
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    raw = {
        "observation.state": state[rows],
        "action": actions[indices],
        "prompt": "",
        **{f"observation.images.cam{i}": image for i in range(3)},
    }
    return transform(raw)


def main(config_name: str = "pi05_fr3_wuji_125ep", batch_size: int = 512):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = training_config.get_config(config_name)
    if not isinstance(config.data, training_config.LeRobotFr3WujiDataConfig):
        raise ValueError("Only LeRobotFr3WujiDataConfig is supported")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.local_dataset_loader is not None or tuple(data_config.action_sequence_keys) != ("action",):
        raise ValueError("Requires standard LeRobot data with the action sequence key 'action'")
    if any(type(t) is not transforms.RepackTransform for t in data_config.repack_transforms.inputs):
        raise ValueError("Unsupported repack transforms")
    if any(
        type(t) not in (fr3_wuji_policy.Fr3WujiInputs, transforms.DeltaActions)
        for t in data_config.data_transforms.inputs
    ):
        raise ValueError("Unsupported transforms: image-dependent transforms cannot use vector-only statistics")
    output = config.assets_dirs / data_config.asset_id
    if (output / "norm_stats.json").exists():
        raise FileExistsError(f"Norm stats already exist at {output}; refusing to overwrite")
    dataset_specs = data_config.lerobot_datasets or (training_config.LeRobotDatasetSpec(repo_id=data_config.repo_id),)
    sources = []
    for spec in dataset_specs:
        root = HF_LEROBOT_HOME / spec.repo_id
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
        sources.append((spec, root, info, episodes))
    transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    reference = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    totals = {key: np.zeros(config.model.action_dim, dtype=np.float64) for key in stats}
    squares = {key: np.zeros(config.model.action_dim, dtype=np.float64) for key in stats}
    counts = dict.fromkeys(stats, 0)
    offset = 0
    checked = 0
    for spec, root, info, episodes in sources:
        source_offset = 0
        for episode in tqdm.tqdm(episodes, desc=f"Vector norm ({spec.repo_id})"):
            index, length = episode["episode_index"], episode["length"]
            path = root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
            table = pq.read_table(path, columns=["observation.state", "action"])
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            assert len(table) == length
            rows = np.array([0, length // 2, length - 1])
            actual = transform_vectors(state, actions, rows, config.model.action_horizon, transform)
            for i, row in enumerate(rows):
                raw = reference[offset + source_offset + int(row)]
                assert int(raw["episode_index"]) == index
                expected = transform(raw)
                for key in stats:
                    np.testing.assert_array_equal(actual[key][i], expected[key])
                checked += 1
            for start in range(0, length, batch_size):
                rows = np.arange(start, min(start + batch_size, length))
                batch = transform_vectors(state, actions, rows, config.model.action_horizon, transform)
                for key, running_stats in stats.items():
                    values = np.asarray(batch[key], dtype=np.float64).reshape(-1, config.model.action_dim)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite values in {spec.repo_id} episode {index}, {key}")
                    running_stats.update(values)
                    totals[key] += values.sum(axis=0)
                    squares[key] += (values**2).sum(axis=0)
                    counts[key] += len(values)
            source_offset += length
        assert source_offset == info["total_frames"]
        offset += source_offset
    assert offset == sum(source[2]["total_frames"] for source in sources) == len(reference)
    assert counts == {"state": offset, "actions": offset * config.model.action_horizon}
    result = {key: value.get_statistics() for key, value in stats.items()}
    for key, value in result.items():
        mean = totals[key] / counts[key]
        std = np.sqrt(np.maximum(0, squares[key] / counts[key] - mean**2))
        np.testing.assert_allclose(value.mean, mean, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(value.std, std, atol=1e-9, rtol=1e-9)
        assert all(np.isfinite(getattr(value, field)).all() for field in ("mean", "std", "q01", "q99"))
        assert np.all(value.std > 0)
        assert np.all(value.q99 > value.q01)
    normalize.save(output, result)
    (output / "norm_metadata.json").write_text(
        json.dumps(
            {
                "config_name": config_name,
                "repo_id": data_config.repo_id,
                "datasets": [
                    {
                        "repo_id": spec.repo_id,
                        "training_weight": spec.weight,
                        "episodes": len(episodes),
                        "frames": info["total_frames"],
                    }
                    for spec, _, info, episodes in sources
                ],
                "episodes": sum(len(source[3]) for source in sources),
                "frames": offset,
                "action_horizon": config.model.action_horizon,
                "vector_counts": counts,
                "lerobot_comparison_samples": checked,
                "accumulation_dtype": "float64",
                "sampling": (
                    "all source frames equally weighted for norm (training weights ignored); "
                    "forward action windows padded within each episode"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: {offset} frames, {checked} LeRobot sample comparisons; norm saved to {output}")


if __name__ == "__main__":
    tyro.cli(main)
