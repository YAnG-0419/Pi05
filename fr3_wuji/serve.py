"""Compatibility service entry; current model adapters live in deployment.models."""
import sys
from fr3_wuji.observation import PROMPT  # noqa: F401 -- legacy smoke client import
from deployment.models.joint54 import checkpoint_contract
from deployment.server import main as serve


def check_checkpoint(checkpoint):
    from pathlib import Path
    metadata, _ = checkpoint_contract(checkpoint)
    return Path(metadata['checkpoint'])


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    # Preserve the old standalone diagnostic server's port.
    if '--port' not in args and not any(arg.startswith('--port=') for arg in args):
        args.extend(['--port', '8000'])
    serve(args, default_profile='slow54')


if __name__ == '__main__':
    main()
