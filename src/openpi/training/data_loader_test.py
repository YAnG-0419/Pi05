import dataclasses

import jax
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_weighted_concat_dataset_sampling():
    dataset = _data_loader.WeightedConcatDataset(
        [list(range(65)), list(range(60))],
        [0.4, 0.6],
    )

    assert len(dataset) == 125
    np.testing.assert_allclose(dataset.sample_weights[:65].sum(), 0.4)
    np.testing.assert_allclose(dataset.sample_weights[65:].sum(), 0.6)

    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(dataset.sample_weights),
        num_samples=20_000,
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )
    sampled = np.fromiter(sampler, dtype=np.int64)
    np.testing.assert_allclose(np.mean(sampled < 65), 0.4, atol=0.015)


def test_left_enhancement_config_and_sampling():
    config = _config.get_config("pi05_fr3_wuji_left_enhance")
    base = _config.get_config("pi05_fr3_wuji")
    specs = config.data.lerobot_datasets

    assert config.model == base.model
    assert config.weight_loader == base.weight_loader
    assert config.batch_size == 8
    assert config.num_train_steps == 20_000
    assert config.data.repo_id == config.data.assets.asset_id
    np.testing.assert_allclose([spec.weight for spec in specs], [8 / 15, 4 / 15, 2 / 15, 1 / 15])
    assert config.data.base_config.video_backend == "pyav"

    # Clip duration can be much shorter than complete episodes; each dataset
    # still receives its configured share of draws, independent of frame count.
    dataset = _data_loader.WeightedConcatDataset(
        [range(300_000), range(4_000)], [sum(spec.weight for spec in specs[:2]), sum(spec.weight for spec in specs[2:])]
    )
    np.testing.assert_allclose(dataset.sample_weights[:300_000].sum(), 0.8)
    np.testing.assert_allclose(dataset.sample_weights[300_000:].sum(), 0.2)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(dataset.sample_weights),
        num_samples=20_000,
        replacement=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    sampled = np.fromiter(sampler, dtype=np.int64)
    np.testing.assert_allclose(np.mean(sampled >= 300_000), 0.2, atol=0.015)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
