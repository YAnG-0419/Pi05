"""Read checkpoint-owned normalization without importing training code."""

import logging
from pathlib import Path

from openpi.shared import normalize


def load_norm_stats(assets_dir: Path | str, asset_id: str) -> dict[str, normalize.NormStats]:
    path = Path(assets_dir) / asset_id
    stats = normalize.load(path)
    logging.info("Loaded norm stats from %s", path)
    return stats
