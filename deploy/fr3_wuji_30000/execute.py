"""30000-specific entry reusing the existing slow executor without changing it."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.fr3_wuji_30000.serve import checkpoint_contract


def main():
    from experiments.weight_motion_eval.oneshot import deploy
    # This binding affects only this dedicated process. The old module on disk
    # and all old entry points retain their original checkpoint contract.
    from deploy.fr3_wuji_30000.planning_profile import install
    install()
    deploy.checkpoint_contract = checkpoint_contract
    args = sys.argv[1:]
    if '--checkpoint' not in args and not any(a.startswith('--checkpoint=') for a in args):
        args.extend(['--checkpoint', str(ROOT / 'checkpoints/30000')])
    if '--uri' not in args and not any(a.startswith('--uri=') for a in args):
        args.extend(['--uri', 'ws://127.0.0.1:8002'])
    sys.argv = [sys.argv[0], *args]
    deploy.main()


if __name__ == '__main__':
    main()
