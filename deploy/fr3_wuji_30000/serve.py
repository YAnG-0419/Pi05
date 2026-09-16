"""Independent 64D checkpoint server with a 54D FR3/Wuji hardware contract."""
import argparse
import dataclasses
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ('Pick up a tomato truss with the right hand, then pick a cherry tomato '
          'with the left hand and place it in the left basket.')


def check_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not (path / 'params/manifest.ocdbt').is_file():
        raise ValueError('Missing complete JAX params directory')
    files = list((path / 'assets').rglob('norm_stats.json'))
    if len(files) != 1:
        raise ValueError('Expected exactly one checkpoint-owned norm_stats.json')
    stats = json.loads(files[0].read_text())['norm_stats']
    for key in ('state', 'actions'):
        for field in ('mean', 'std', 'q01', 'q99'):
            values = np.asarray(stats[key][field], dtype=float)
            if values.shape != (64,) or not np.isfinite(values).all():
                raise ValueError(f'{key}.{field} must contain 64 finite values')
            if np.any(values[54:] != 0):
                raise ValueError(f'{key}.{field}: padding dimensions must be zero')
        if np.any(np.asarray(stats[key]['std']) < 0):
            raise ValueError('Negative standard deviation')
        if np.any(np.asarray(stats[key]['q01']) > stats[key]['q99']):
            raise ValueError('Invalid quantile order')
    return path, files[0]


@dataclasses.dataclass(frozen=True)
class ModelOrderedState:
    def __call__(self, data):
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (54,) or not np.isfinite(state).all():
            raise ValueError('Expected one finite 54D physical state')
        # Hardware: LA7, LH20, RA7, RH20. Checkpoint: LA7, RA7, LH20, RH20, zero10.
        ordered = np.concatenate((state[:7], state[27:34], state[7:27], state[34:54]))
        return {**data, 'state': np.pad(ordered, (0, 10))}


@dataclasses.dataclass(frozen=True)
class PhysicalAbsoluteActions:
    def __call__(self, data):
        # Called AFTER checkpoint-owned unnormalization. These are already
        # absolute positions: do not add the current robot state again.
        actions = np.asarray(data['actions'])
        if actions.shape != (50, 64) or not np.isfinite(actions).all():
            raise ValueError('Expected finite model actions of shape (50, 64)')
        physical = np.concatenate(
            (actions[:, :7], actions[:, 14:34], actions[:, 7:14], actions[:, 34:54]), axis=-1)
        return {'actions': physical}


@dataclasses.dataclass(frozen=True)
class AdaptData:
    base: object

    def create(self, assets_dirs, model_config):
        from openpi import transforms
        data = dataclasses.replace(self.base, extra_delta_transform=False).create(assets_dirs, model_config)
        # Reorder and pad BEFORE normalization/tokenization. Keep the image mapping.
        group = transforms.Group(
            inputs=(*data.data_transforms.inputs, ModelOrderedState()),
            outputs=(PhysicalAbsoluteActions(),),
        )
        return dataclasses.replace(data, data_transforms=group)


class CheckedPolicy:
    def __init__(self, policy, metadata):
        self.policy, self.metadata = policy, metadata

    def infer(self, obs, **kwargs):
        state = np.asarray(obs['observation/state'])
        if state.shape != (54,) or not np.isfinite(state).all():
            raise ValueError('Expected finite observation/state of shape (54,)')
        result = self.policy.infer(obs, **kwargs)
        actions = np.asarray(result['actions'])
        if actions.shape != (50, 54) or not np.isfinite(actions).all():
            raise ValueError('Expected finite physical actions of shape (50, 54)')
        return result


def checkpoint_contract(checkpoint):
    checkpoint, stats_file = check_checkpoint(checkpoint)
    def digest_file(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'config': 'pi05_fr3_wuji_30000_64to54',
        'adapter_revision': 2,
        'model_state_order': ['left_arm_7', 'right_arm_7', 'left_hand_20', 'right_hand_20', 'padding_10'],
        'model_action_order': ['left_arm_7', 'right_arm_7', 'left_hand_20', 'right_hand_20', 'padding_10'],
        'model_action_dim': 64, 'action_dim': 54, 'action_horizon': 50,
        'action_order': ['left_arm_7', 'left_hand_20', 'right_arm_7', 'right_hand_20'],
        'checkpoint': str(checkpoint), 'hardware_output': False, 'padding_dim': 10,
        'checkpoint_manifest_sha256': digest_file(checkpoint / 'params/manifest.ocdbt'),
        'normalization_sha256': digest_file(stats_file),
        'action_representation': 'absolute_joint_positions',
    }, stats_file


def create_policy(checkpoint, stats_file, prompt):
    from openpi.training import config, checkpoints
    from openpi.policies import policy_config
    base = config.get_config('pi05_fr3_wuji')
    adapted = dataclasses.replace(
        base, name='pi05_fr3_wuji_30000_64to54',
        model=dataclasses.replace(base.model, action_dim=64),
        data=AdaptData(base.data),
    )
    stats = checkpoints.load_norm_stats(
        checkpoint / 'assets', str(stats_file.parent.relative_to(checkpoint / 'assets')))
    policy = policy_config.create_trained_policy(
        adapted, checkpoint, norm_stats=stats, default_prompt=prompt)
    # Unnormalize 64D absolute positions, reorder to the hardware contract, discard padding.
    metadata, _ = checkpoint_contract(checkpoint)
    return CheckedPolicy(policy, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/30000')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true', help='Load and infer fake observations; never serve or connect hardware')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Invalid port')
    checkpoint, stats_file = check_checkpoint(args.checkpoint)
    print(f'64D checkpoint statistics OK: {stats_file}', flush=True)
    if args.check_only:
        return
    import jax
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('CUDA GPU required')
    policy = create_policy(checkpoint, stats_file, args.prompt)
    obs = {'observation/state': np.zeros(54, np.float32), 'prompt': args.prompt}
    for key in ('observation/image', 'observation/left_wrist_image', 'observation/right_wrist_image'):
        obs[key] = np.zeros((224, 224, 3), np.uint8)
    result = policy.infer(obs)
    print(f'Warmup OK: actions={result["actions"].shape}', flush=True)
    policy.metadata['warmed_up'] = True
    if args.smoke_only:
        return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    print(f'Serving 54D hardware contract at ws://{args.host}:{args.port}', flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == '__main__':
    main()
