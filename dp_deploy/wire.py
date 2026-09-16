"""Bounded NPZ-over-HTTP transport; never pickle over the network."""

import io
import json
import urllib.request
import zipfile
import numpy as np

MAX_BYTES = 32 * 1024 * 1024


def pack(arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def unpack(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Oversized payload")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(n.file_size for n in z.infolist()) > MAX_BYTES:
            raise ValueError("Oversized decoded arrays")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def request(url, arrays=None, timeout=3):
    req = urllib.request.Request(
        url,
        data=None if arrays is None else pack(arrays),
        headers={"Content-Type": "application/octet-stream"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Oversized response")
        return (
            json.loads(data)
            if r.headers.get_content_type() == "application/json"
            else unpack(data)
        )
