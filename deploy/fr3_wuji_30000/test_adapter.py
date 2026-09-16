"""CPU-only regression checks; no robot, ROS, SDK or model inference."""
import dataclasses
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.fr3_wuji_30000 import serve
from openpi import transforms
from openpi.training import config, checkpoints


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = config.get_config('pi05_fr3_wuji')
        cls.model = dataclasses.replace(cls.base.model, action_dim=64)
        cls.data = serve.AdaptData(cls.base.data).create(cls.base.assets_dirs, cls.model)
        cls.checkpoint, stats_file = serve.check_checkpoint(ROOT / 'checkpoints/30000')
        cls.stats = checkpoints.load_norm_stats(cls.checkpoint / 'assets',
            str(stats_file.parent.relative_to(cls.checkpoint / 'assets')))

    def observation(self, state):
        obs = {'observation/state': state, 'prompt': serve.PROMPT}
        for name in ('image', 'left_wrist_image', 'right_wrist_image'):
            obs['observation/' + name] = np.zeros((224, 224, 3), np.uint8)
        return obs

    def test_physical_to_model_and_back(self):
        physical = np.arange(54, dtype=np.float32)
        data = transforms.compose(self.data.data_transforms.inputs)(self.observation(physical))
        expected = np.r_[0:7, 27:34, 7:27, 34:54, np.zeros(10)]
        np.testing.assert_array_equal(data['state'], expected)
        # Nonzero, unrelated state must not be added to absolute predictions.
        actions = np.tile(expected, (50, 1))
        actions[:, 54:] = 99
        result = transforms.compose(self.data.data_transforms.outputs)(
            {'state': np.full(64, 1000.), 'actions': actions})
        np.testing.assert_array_equal(result['actions'], np.tile(physical, (50, 1)))

    def test_full_normalization_output_chain(self):
        state = np.arange(54, dtype=np.float32) / 100
        x = transforms.compose(self.data.data_transforms.inputs)(self.observation(state))
        normalized = transforms.Normalize(self.stats, use_quantiles=self.data.use_quantile_norm)(x)
        tokens = transforms.compose(self.data.model_transforms.inputs)(normalized)
        self.assertEqual(tokens['state'].shape, (64,))
        self.assertTrue(np.isfinite(tokens['state']).all())
        # Quantile normalized 0 is the quantile midpoint, already absolute.
        y = transforms.Unnormalize(self.stats, use_quantiles=self.data.use_quantile_norm)(
            {'state': tokens['state'], 'actions': np.zeros((50, 64))})
        absolute = y['actions'].copy()
        result = transforms.compose(self.data.data_transforms.outputs)(y)['actions']
        np.testing.assert_allclose(result[:, :7], absolute[:, :7])
        np.testing.assert_allclose(result[:, 7:27], absolute[:, 14:34])
        np.testing.assert_allclose(result[:, 27:34], absolute[:, 7:14])
        self.assertEqual(result.shape, (50, 54))

    def test_invalid_inputs(self):
        for state in (np.zeros(64), np.full(54, np.nan)):
            with self.assertRaises(ValueError):
                serve.ModelOrderedState()({'state': state})
        for actions in (np.zeros((50,54)), np.zeros((1,64)), np.full((50,64),np.nan)):
            with self.assertRaises(ValueError):
                serve.PhysicalAbsoluteActions()({'actions': actions})

    def test_stale_service_contract_rejected(self):
        expected, _ = serve.checkpoint_contract(self.checkpoint)
        old = {k: v for k,v in expected.items() if k not in ('adapter_revision','model_state_order','model_action_order')}
        old['action_representation'] = 'absolute_joint_positions_after_arm_delta_transform'
        self.assertTrue(any(old.get(k) != v for k,v in expected.items()))
        self.assertEqual(expected['adapter_revision'], 2)
        self.assertTrue(self.base.data.extra_delta_transform)  # shared config unchanged
        self.assertEqual(self.base.model.action_dim, 54)


if __name__ == '__main__':
    unittest.main()
