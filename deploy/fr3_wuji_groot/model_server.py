"""Serve a verified dual-FR3/Wuji GR00T checkpoint without any hardware access.

Run this module with the Isaac-GR00T Python 3.12 environment while making both
repositories importable. The wire contract uses Pi05's physical joint order;
GR00T-specific ordering remains inside this process.
"""

from __future__ import annotations

import argparse
import contextlib
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from .contract import ACTION_DIM
from .contract import ACTION_HORIZON
from .contract import ACTION_HZ
from .contract import ACTION_ORDER
from .contract import MAX_BODY_BYTES
from .contract import PHYSICAL_SLICES
from .contract import PROTOCOL
from .contract import checkpoint_manifest
from .contract import finite_vector
from .contract import groot_to_physical
from .contract import semantic_json_sha256
from .contract import validate_images
from .model_cache import configure_model_cache


def checkpoint_contract(checkpoint: Path) -> tuple[dict, list[dict[str, object]]]:
    checkpoint = checkpoint.expanduser().resolve()
    statistics_sha256 = semantic_json_sha256(checkpoint / "statistics.json")
    recipe = json.loads((checkpoint / "inference_recipe.json").read_text())
    verification = json.loads((checkpoint / "reload_verification.json").read_text())
    if recipe.get("statistics_sha256") != statistics_sha256:
        raise ValueError("GR00T inference recipe does not match checkpoint statistics")
    if (
        verification.get("status") != "verified"
        or verification.get("statistics_sha256") != statistics_sha256
        or verification.get("output_shape") != [ACTION_HORIZON, ACTION_DIM]
        or verification.get("max_difference") != 0.0
    ):
        raise ValueError("GR00T checkpoint has not passed exact reload verification")
    if recipe.get("action_head_dtype") != "float32" or recipe.get("padding_matching") is not True:
        raise ValueError("GR00T checkpoint inference recipe is incompatible with TomatoPolicy")
    model_sha256, manifest = checkpoint_manifest(checkpoint)
    metadata = {
        "protocol": PROTOCOL,
        "backend": "groot_n1d7",
        "checkpoint": str(checkpoint),
        "model_manifest_sha256": model_sha256,
        "statistics_sha256": statistics_sha256,
        "action_shape": [ACTION_HORIZON, ACTION_DIM],
        "action_hz": ACTION_HZ,
        "action_order": ACTION_ORDER,
        "joint_units": "radians",
        "action_semantics": "absolute_joint_targets",
        "reload_verified": True,
        "warmed_up": False,
        "hardware_output": False,
        "instruction": recipe.get("full_task_instruction"),
        "action_head_dtype": recipe.get("action_head_dtype"),
        "backbone_dtype": verification.get("backbone_dtype"),
        "padding_matching": True,
        "inference_timesteps": recipe.get("num_inference_timesteps"),
    }
    return metadata, manifest


class GrootInference:
    def __init__(self, policy):
        self.policy = policy

    def predict(self, arrays: dict[str, object]) -> np.ndarray:
        state = finite_vector(arrays["state"], label="physical state")
        images = validate_images({name: arrays[name] for name in ("head", "left_wrist", "right_wrist")})
        instruction = np.asarray(arrays["instruction"])
        if instruction.shape != () or instruction.dtype.kind not in {"U", "S"}:
            raise ValueError("instruction must be one scalar string")
        prompt = str(instruction.item())
        if not prompt.strip():
            raise ValueError("instruction must not be empty")
        joints = {name: state[index] for name, index in PHYSICAL_SLICES.items()}
        # TomatoPolicy returns GR00T order: left arm, right arm, left hand, right hand.
        return groot_to_physical(self.policy.predict(images, joints, prompt))


def read_npz(body: bytes) -> dict[str, np.ndarray]:
    if not 0 < len(body) <= MAX_BODY_BYTES:
        raise ValueError("Request body is empty or exceeds the configured limit")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        members = archive.infolist()
        if len({item.filename for item in members}) != len(members):
            raise ValueError("Duplicate NPZ members are not accepted")
        if sum(item.file_size for item in members) > MAX_BODY_BYTES:
            raise ValueError("Uncompressed request exceeds the configured limit")
    with np.load(io.BytesIO(body), allow_pickle=False) as payload:
        required = {"state", "instruction", "head", "left_wrist", "right_wrist"}
        if set(payload.files) != required:
            raise ValueError(f"Request fields must be exactly {sorted(required)}")
        return {key: payload[key] for key in required}


def response_npz(actions: np.ndarray, inference_seconds: float) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        actions=actions,
        inference_seconds=np.asarray(inference_seconds, dtype=np.float64),
    )
    return output.getvalue()


def make_handler(inference: GrootInference, metadata: dict):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self.reply(200, b'{"status":"ok"}')
            elif self.path == "/metadata":
                self.reply(200, json.dumps(metadata, sort_keys=True).encode())
            else:
                self.reply(404, b'{"error":"Unknown endpoint"}')

        def do_POST(self) -> None:
            try:
                if self.path != "/action" or self.headers.get("X-Protocol") != PROTOCOL:
                    raise ValueError("Wrong endpoint or protocol")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise ValueError("Invalid Content-Length")
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("Truncated request body")
                arrays = read_npz(body)
                started = time.perf_counter()
                actions = inference.predict(arrays)
                elapsed = time.perf_counter() - started
                self.reply(200, response_npz(actions, elapsed), "application/x-npz")
                print("ACTION", json.dumps({"shape": list(actions.shape), "seconds": elapsed}), flush=True)
            except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
                self.reply(400, json.dumps({"error": str(error)}).encode())
            except Exception as error:
                print("INFERENCE_ERROR", repr(error), flush=True)
                self.reply(500, json.dumps({"error": type(error).__name__}).encode())

        def log_message(self, format, *args):
            print("GR00T_HTTP", format % args, flush=True)

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-cache",
        type=Path,
        help="Portable models_http directory; required unless --check-only is used",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        metadata, manifest = checkpoint_contract(args.checkpoint)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.model_cache is not None:
        try:
            base_models = configure_model_cache(args.model_cache)
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            parser.error(str(error))
        metadata["base_model_revisions"] = {model_id: details["revision"] for model_id, details in base_models.items()}
    print(json.dumps({**metadata, "manifest": manifest}, indent=2), flush=True)
    if args.check_only:
        return
    if args.model_cache is None:
        parser.error("--model-cache is required when loading the GR00T policy")

    # Deliberately lazy: --check-only and Pi05-side tests must not import the
    # Python 3.12 GR00T stack into the Python 3.11 robot environment.
    from examples.dual_franka_wuji.tomato_policy import TomatoPolicy  # noqa: PLC0415

    policy = TomatoPolicy(args.checkpoint)
    inference = GrootInference(policy)
    images = {name: np.zeros((224, 224, 3), dtype=np.uint8) for name in ("head", "left_wrist", "right_wrist")}
    inference.predict({"state": np.zeros(ACTION_DIM), "instruction": np.asarray(metadata["instruction"]), **images})
    metadata["warmed_up"] = True
    print(f"GR00T_POLICY_READY http://{args.host}:{args.port}", flush=True)
    with (
        HTTPServer((args.host, args.port), make_handler(inference, metadata)) as server,
        contextlib.suppress(KeyboardInterrupt),
    ):
        server.serve_forever()


if __name__ == "__main__":
    main()
