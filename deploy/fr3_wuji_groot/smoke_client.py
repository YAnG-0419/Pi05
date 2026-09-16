"""Send one synthetic observation to GR00T without importing hardware code."""

import argparse
import json

import numpy as np

from .client import GrootClient
from .contract import ACTION_DIM


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8766")
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    client = GrootClient(
        args.server,
        expected_model_sha256=args.expected_model_sha256,
        timeout=min(args.timeout, 60.0),
    )
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    result = client.infer(
        {
            "observation/state": np.zeros(ACTION_DIM, dtype=np.float32),
            "observation/image": image,
            "observation/left_wrist_image": image,
            "observation/right_wrist_image": image,
        },
        timeout=args.timeout,
    )
    actions = result["actions"]
    print(
        json.dumps(
            {
                "hardware_output": False,
                "model_manifest_sha256": client.metadata["model_manifest_sha256"],
                "base_model_revisions": client.metadata.get("base_model_revisions"),
                "shape": list(actions.shape),
                "finite": bool(np.isfinite(actions).all()),
                "minimum": float(actions.min()),
                "maximum": float(actions.max()),
                "policy_timing": result["policy_timing"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
