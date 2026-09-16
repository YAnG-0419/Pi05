"""Check the isolated inference environment without loading weights or connecting hardware."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-tokenizer", action="store_true", help="Download/validate the Pi05 tokenizer cache.")
    parser.add_argument("--output", type=Path, default=Path("logs/fr3_wuji/environment.json"))
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.policies import policy_config  # noqa: F401 -- verify complete serving import chain
    from openpi.training import config

    devices = jax.devices()
    if not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"GPU unavailable: {devices}")
    value = jax.jit(lambda x: x @ x)(jnp.eye(32)).block_until_ready()
    if float(value.sum()) != 32.0:
        raise RuntimeError("GPU computation returned an unexpected result")
    convolution = jax.jit(
        lambda x, kernel: jax.lax.conv_general_dilated(
            x, kernel, window_strides=(1, 1), padding="VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
    )
    features = convolution(
        jnp.ones((1, 16, 16, 3), dtype=jnp.bfloat16), jnp.ones((3, 3, 3, 8), dtype=jnp.bfloat16)
    ).block_until_ready()
    if not bool(jnp.all(features == 27)):
        raise RuntimeError("GPU BF16 convolution returned an unexpected result")
    model = config.get_config("pi05_fr3_wuji").model
    if (model.action_dim, model.action_horizon) != (54, 50):
        raise RuntimeError("Training configuration differs from this deployment's 54 x 50 contract")
    tokenizer_ready = False
    if args.prepare_tokenizer:
        from openpi.models.tokenizer import PaligemmaTokenizer

        tokens, mask = PaligemmaTokenizer(max_len=256).tokenize("deployment check")
        if tokens.shape != (256,) or not mask.any():
            raise RuntimeError("Tokenizer check failed")
        tokenizer_ready = True
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "devices": [str(device) for device in devices],
        "gpu_computation_passed": True,
        "gpu_bfloat16_convolution_passed": True,
        "tokenizer_ready": tokenizer_ready,
        "config": "pi05_fr3_wuji",
        "action_dim": model.action_dim,
        "action_horizon": model.action_horizon,
        "packages": {
            name: version(name)
            for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "torch", "websockets", "transformers")
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "openpi_data_home": os.environ.get("OPENPI_DATA_HOME"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
