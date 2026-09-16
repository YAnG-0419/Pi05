"""HTTP/NPZ client for the isolated GR00T inference process."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np

from .contract import IMAGE_KEYS
from .contract import MAX_BODY_BYTES
from .contract import PROTOCOL
from .contract import finite_actions
from .contract import finite_vector
from .contract import validate_metadata


class GrootClient:
    def __init__(self, server: str, *, expected_model_sha256: str | None = None, timeout: float = 5.0):
        parsed = urllib.parse.urlsplit(server)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("GR00T server must be an HTTP origin such as http://127.0.0.1:8766")
        if not 0 < timeout <= 60:
            raise ValueError("GR00T timeout must be in (0, 60] seconds")
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.metadata = validate_metadata(self._get_metadata(), expected_model_sha256=expected_model_sha256)

    def _request(self, request: urllib.request.Request, *, timeout: float | None = None) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                size = int(response.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY_BYTES:
                    raise RuntimeError("GR00T response is empty or exceeds the configured limit")
                body = response.read(MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            raise RuntimeError(f"GR00T server rejected the request: HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Cannot reach GR00T server at {self.server}: {error.reason}") from error
        if len(body) != size or len(body) > MAX_BODY_BYTES:
            raise RuntimeError("GR00T response length does not match its bounded Content-Length")
        return body

    def _get_metadata(self) -> object:
        body = self._request(urllib.request.Request(self.server + "/metadata", method="GET"))
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("GR00T metadata response is not valid JSON") from error

    def infer(self, observation: dict, *, timeout: float | None = None) -> dict:
        state = finite_vector(observation.get("observation/state"), label="observation/state")
        # This checkpoint was trained and reload-verified with one full task
        # instruction.  Do not accidentally inherit a Pi05 prompt from the
        # observation pipeline.
        prompt = self.metadata.get("instruction")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("The GR00T server must publish its non-empty checkpoint instruction")
        arrays = {name: observation.get(key) for name, key in IMAGE_KEYS.items()}
        for name, value in arrays.items():
            image = np.asarray(value)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name} observation must be RGB uint8 [height,width,3]")
            arrays[name] = image
        payload = io.BytesIO()
        np.savez_compressed(payload, state=state, instruction=np.asarray(prompt), **arrays)
        data = payload.getvalue()
        if len(data) > MAX_BODY_BYTES:
            raise ValueError("GR00T request exceeds the configured body limit")
        request = urllib.request.Request(
            self.server + "/action",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-npz", "X-Protocol": PROTOCOL},
        )
        body = self._request(request, timeout=timeout)
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                if sum(item.file_size for item in archive.infolist()) > MAX_BODY_BYTES:
                    raise RuntimeError("Uncompressed GR00T response exceeds the configured limit")
            with np.load(io.BytesIO(body), allow_pickle=False) as result:
                actions = finite_actions(result["actions"], order="physical")
                inference_seconds = float(np.asarray(result["inference_seconds"]).item())
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
            raise RuntimeError("GR00T response is not a valid bounded numeric NPZ payload") from error
        if not np.isfinite(inference_seconds) or inference_seconds < 0:
            raise RuntimeError("GR00T server returned invalid inference timing")
        return {"actions": actions, "policy_timing": {"infer_ms": inference_seconds * 1000}}
