"""Regression coverage for removing training without changing inference contracts."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from openpi import transforms
from openpi.inference import config
from openpi.models import tokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_inference_imports_do_not_require_training_or_datasets():
    code = '''
import importlib.abc
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('openpi.training'):
            raise AssertionError('Inference tried to import ' + fullname)
sys.meta_path.insert(0, BlockTraining())
# Optional-package probes may inspect availability; actual imports must fail.
for name in ('lerobot', 'wandb', 'datasets'):
    sys.modules[name] = None
from openpi.policies import policy_config
from openpi.models import pi0, gemma, siglip
from openpi.inference.config import get_config
model = get_config('pi05_fr3_wuji').model
assert (model.action_dim, model.action_horizon, model.max_token_len) == (54, 50, 256)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env={**os.environ, 'JAX_PLATFORMS': 'cpu'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_19999_restores_arm_deltas_and_preserves_absolute_hands(monkeypatch):
    monkeypatch.setattr(tokenizer, 'PaligemmaTokenizer', lambda *a: object())
    cfg = config.get_config('pi05_fr3_wuji')
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data.use_quantile_norm
    assert cfg.model.paligemma_variant == 'gemma_2b_lora'
    assert cfg.model.action_expert_variant == 'gemma_300m_lora'
    state = np.linspace(-1, 1, 54, dtype=np.float32)
    actions = np.full((50, 54), 0.25, np.float32)
    result = transforms.compose(data.data_transforms.outputs)({'state': state, 'actions': actions.copy()})['actions']
    arm = np.r_[0:7, 27:34]
    hand = np.r_[7:27, 34:54]
    np.testing.assert_allclose(result[:, arm], actions[:, arm] + state[arm])
    np.testing.assert_array_equal(result[:, hand], actions[:, hand])


def test_shell_wrappers_locate_this_project_outside_cwd():
    result = subprocess.run(['bash', str(ROOT/'fr3_wuji/run.sh'), '-c',
        'import openpi, openpi_client; from fr3_wuji.paths import ROOT; print(ROOT); print(openpi.__path__[0]); print(openpi_client.__path__[0])'], cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == [str(ROOT), str(ROOT/'src/openpi'), str(ROOT/'packages/openpi-client/src/openpi_client')]


def test_configuration_rejects_removed_training_presets():
    with pytest.raises(ValueError, match='Unsupported deployment config'):
        config.get_config('pi05_fr3_wuji_269ep')
    assert not (ROOT/'src/openpi/training').exists()
