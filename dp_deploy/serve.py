"""GPU inference service. Does not import ROS or connect to hardware."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import numpy as np
from .config import load_config
from .wire import MAX_BYTES, pack, unpack


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", default="reports/model-smoke.json")
    a = p.parse_args()
    import torch

    torch.set_num_threads(4)
    from .checkpoint import Policy
    from .contract import OBS_SHAPES

    c = load_config()
    began = time.monotonic()
    policy = Policy(a.checkpoint or c["checkpoint"], a.device)
    dummy = {k: np.zeros((2, *s), np.float32) for k, s in OBS_SHAPES.items()}
    times = []
    for _ in range(3):
        start = time.monotonic()
        result = policy.infer(dummy)
        times.append(time.monotonic() - start)
    policy.metadata.update(
        warmed_up=True, warmup_seconds=times, load_and_warmup_s=time.monotonic() - began
    )
    print(json.dumps(policy.metadata, indent=2), flush=True)
    if a.smoke:
        from pathlib import Path

        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **policy.metadata,
                    "hardware_output": False,
                    "synthetic_observations": True,
                    "output_shape": list(result["action"].shape),
                },
                indent=2,
            )
            + "\n"
        )
        return
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/metadata":
                self.send(404, b"unknown route", "text/plain")
                return
            self.send(200, json.dumps(policy.metadata).encode(), "application/json")

        def do_POST(self):
            if self.path != "/infer":
                self.send(404, b"unknown route", "text/plain")
                return
            if not lock.acquire(blocking=False):
                self.send(409, b"policy busy", "text/plain")
                return
            try:
                self.connection.settimeout(5)
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= MAX_BYTES:
                    raise ValueError("Invalid request size")
                raw = self.rfile.read(n)
                if len(raw) != n:
                    raise ValueError("Truncated request")
                obs = unpack(raw)
                began = time.monotonic()
                out = policy.infer(obs)
                out["inference_seconds"] = np.array(time.monotonic() - began)
                self.send(200, pack(out), "application/octet-stream")
            except Exception as e:
                self.send(
                    400, json.dumps({"error": str(e)}).encode(), "application/json"
                )
            finally:
                lock.release()

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Policy ready: http://{a.host}:{a.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
