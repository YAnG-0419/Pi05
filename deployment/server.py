"""Shared inference server. Adapters only convert model inputs/outputs; no device access."""
import argparse
import json
from pathlib import Path
import numpy as np
from deployment.profiles import MODELS, PROFILES, compose_profile, get_profile
from fr3_wuji.observation import IMAGE_KEYS, IMAGE_SHAPES, PROMPT


def main(argv=None, *, default_profile='slow54'):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--profile', choices=tuple(PROFILES), default=default_profile, help='Legacy combined preset')
    selection.add_argument('--model', choices=tuple(MODELS), help='pi05 checkpoint and adapter selection')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--prompt', default=PROMPT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args(argv)
    profile = compose_profile('slow', args.model) if args.model else get_profile(args.profile)
    port = args.port if args.port is not None else profile.port
    if not 1 <= port <= 65535:
        parser.error('Invalid port')
    adapter = profile.model_adapter()
    metadata, stats_file = adapter.checkpoint_contract(args.checkpoint or profile.checkpoint)
    if args.check_only:
        print(json.dumps({'profile': profile.name, 'adapter': profile.adapter, **metadata}, indent=2))
        return
    import jax
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('GPU inference required')
    policy = adapter.create_policy(Path(metadata['checkpoint']), stats_file, args.prompt)
    observation = {'observation/state': np.zeros(54, np.float32), 'prompt': args.prompt}
    observation.update({key: np.zeros(IMAGE_SHAPES[name], np.uint8) for name, key in IMAGE_KEYS.items()})
    # Synthetic observations only validate inference and are never sent to devices.
    result = policy.infer(observation)
    actions = np.asarray(result['actions'])
    if actions.shape != (50, 54) or not np.isfinite(actions).all():
        raise ValueError(f'Expected finite (50, 54) hardware actions, got {actions.shape}')
    metadata['warmed_up'] = True
    print(f'Warmup OK: profile={profile.name}, actions={actions.shape}', flush=True)
    if args.smoke_only:
        return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    print(f'Serving {profile.adapter} at ws://{args.host}:{port}', flush=True)
    WebsocketPolicyServer(policy, host=args.host, port=port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
