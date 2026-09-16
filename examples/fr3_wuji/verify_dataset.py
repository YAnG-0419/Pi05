"""Check a converted FR3/Wuji dataset against the original captures.

Vectors must match exactly. Images are re-encoded with a lossy codec, so alignment is
checked instead: a converted frame has to be closer to its own source frame than to any
neighbouring source frame. Near-static frames cannot tell neighbours apart, so the
comparison is restricted to the highest-motion rows. cam0 records at 20fps while the
dataset runs at 30fps, which makes some neighbouring source frames identical; those are
excluded from the comparison rather than counted as competitors.

Example:
    uv run python -m examples.fr3_wuji.verify_dataset --repo-id fr3_wuji/tomato
"""

import dataclasses
import pathlib

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from examples.fr3_wuji.convert_dataset import CAMERA_KEYS
from examples.fr3_wuji.convert_dataset import SOURCE_LOADER_FILENAME
from examples.fr3_wuji.convert_dataset import VECTOR_KEYS
from examples.fr3_wuji.convert_dataset import _find_captures
from examples.fr3_wuji.convert_dataset import _load_source_loader
from examples.fr3_wuji.convert_dataset import _make_capture_reader


@dataclasses.dataclass
class Args:
    repo_id: str = "fr3_wuji/tomato"
    src_root: pathlib.Path = pathlib.Path("/home/descfly/datasets")
    root: pathlib.Path | None = None
    # Number of episodes to check, taken from the start of the dataset.
    episodes: int = 2
    # Number of frames sampled per episode.
    samples: int = 6
    # How far to look on each side when testing frame alignment.
    neighbourhood: int = 3
    # Rows probed while searching for motion, per sample.
    probes_per_sample: int = 8
    # A neighbour only counts as a competitor if it differs from the own source frame by
    # this multiple of the codec reconstruction error.
    discriminability: float = 2.0


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """LeRobot returns float CHW in [0, 1]; the captures return uint8 CHW."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image.transpose(1, 2, 0) if image.shape[0] == 3 else image


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean())


def _pick_moving_rows(source, key: str, episode_length: int, args: Args) -> list[int]:
    """Rank rows by how much the scene changes, so neighbours are distinguishable."""
    first = args.neighbourhood
    last = episode_length - args.neighbourhood - 1
    stride = max(1, (last - first) // max(1, args.samples * args.probes_per_sample))
    scored = []
    for row in range(first, last, stride):
        motion = _distance(_to_hwc_uint8(source[row][key].numpy()), _to_hwc_uint8(source[row + 1][key].numpy()))
        scored.append((motion, row))
    scored.sort(reverse=True)
    return [row for _, row in scored[: args.samples]]


def main(args: Args) -> None:
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    captures = _find_captures(args.src_root, max_episodes=args.episodes)
    capture_reader = _make_capture_reader(_load_source_loader(captures[0] / SOURCE_LOADER_FILENAME))

    print(f"{dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"tasks: {dataset.meta.tasks}")
    for key in (*VECTOR_KEYS, *CAMERA_KEYS):
        print(f"  {key}: {dataset.meta.features[key]['shape']}")

    offset = 0
    for episode_position, capture in enumerate(captures):
        source = capture_reader(capture, vector_keys=VECTOR_KEYS, video_keys=CAMERA_KEYS)
        try:
            episode_length = dataset.meta.episodes[episode_position]["length"]
            if episode_length != len(source):
                raise AssertionError(f"{capture.name}: length {episode_length} != source {len(source)}")

            worst_vector = 0.0
            for row in np.linspace(0, episode_length - 1, args.samples).astype(int):
                item = dataset[offset + int(row)]
                expected = source[int(row)]
                if item["episode_index"].item() != episode_position:
                    raise AssertionError(f"{capture.name}: episode index mismatch at row {row}")
                for key in VECTOR_KEYS:
                    diff = float(np.abs(item[key].numpy() - expected[key].numpy()).max())
                    worst_vector = max(worst_vector, diff)

            worst_margin = None
            comparisons = 0
            failures = []
            for key in CAMERA_KEYS:
                # Each camera moves at its own moments, so rank rows per camera.
                for row in _pick_moving_rows(source, key, episode_length, args):
                    own = _to_hwc_uint8(source[row][key].numpy())
                    converted = _to_hwc_uint8(dataset[offset + row][key].numpy())
                    own_distance = _distance(converted, own)

                    competitors = {}
                    for shift in range(-args.neighbourhood, args.neighbourhood + 1):
                        if shift == 0:
                            continue
                        candidate = _to_hwc_uint8(source[row + shift][key].numpy())
                        # A neighbour that barely differs from the own frame (cam0 repeats
                        # frames at 20fps, and static scenes look alike) cannot decide anything.
                        if _distance(candidate, own) > args.discriminability * max(own_distance, 1e-6):
                            competitors[shift] = _distance(converted, candidate)
                    if not competitors:
                        continue

                    comparisons += 1
                    closest = min(competitors, key=competitors.get)
                    margin = competitors[closest] - own_distance
                    worst_margin = margin if worst_margin is None else min(worst_margin, margin)
                    if margin <= 0:
                        failures.append((key, row, own_distance, closest, competitors[closest]))

            status = "OK" if not failures and worst_vector == 0.0 else "FAIL"
            margin_text = "n/a" if worst_margin is None else f"{worst_margin:.2f}"
            print(
                f"[{status}] {capture.name}: {episode_length} frames, "
                f"max vector diff {worst_vector:g}, "
                f"{comparisons} alignment checks, min margin {margin_text}"
            )
            for key, row, own_distance, closest, closest_distance in failures:
                print(
                    f"    {key} row {row}: own frame {own_distance:.2f} is not closer than "
                    f"offset {closest} at {closest_distance:.2f}"
                )
            if status == "FAIL":
                raise AssertionError(f"{capture.name} failed verification")

            offset += episode_length
        finally:
            source.close()

    print("All checks passed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
