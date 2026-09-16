"""Send synthetic observations to an inference server, with bounded waits. No hardware I/O."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from openpi_client import msgpack_numpy
from serve import PROMPT
from websockets.sync.client import connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300, help="Seconds per response, including first JIT warmup.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("logs/fr3_wuji/smoke.json"))
    args = parser.parse_args()
    if args.count < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("count and timeout must be positive and finite")
    obs = {
        "observation/state": np.zeros(54, dtype=np.float32),
        "observation/image": np.zeros((400, 640, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "prompt": PROMPT,
    }
    report = {"uri": args.uri, "synthetic_observations": True, "runs": []}
    packer = msgpack_numpy.Packer()
    # Explicitly bypass shell proxies for this local/LAN connection (websockets 15).
    with connect(args.uri, compression=None, max_size=None, open_timeout=10, close_timeout=5, proxy=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv(timeout=args.timeout))
        print(f"Server metadata: {metadata}", flush=True)
        for index in range(args.count):
            start = time.monotonic()
            ws.send(packer.pack(obs))
            payload = ws.recv(timeout=args.timeout)
            if isinstance(payload, str):
                raise RuntimeError(f"Server error: {payload}")
            result = msgpack_numpy.unpackb(payload)
            actions = np.asarray(result["actions"])
            if actions.shape != (50, 54) or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite (50, 54) actions, got {actions.shape}")
            run = {
                "index": index,
                "warmup": index == 0,
                "seconds": time.monotonic() - start,
                "shape": list(actions.shape),
                "finite": True,
                "server_timing": result.get("server_timing", {}),
            }
            report["runs"].append(run)
            print(json.dumps(run), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Synthetic inference passed. Report: {args.output}. These actions must not be sent to hardware.")


if __name__ == "__main__":
    main()
