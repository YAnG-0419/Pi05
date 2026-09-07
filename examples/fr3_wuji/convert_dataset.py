"""Merge the per-episode FR3/Wuji captures into a single standard LeRobot dataset.

Each source capture is a single-episode LeRobot v2.0 directory whose per-camera video
references carry their own time base: cam0 runs at 20fps, cam1/cam2 at 30fps, and all of
them are offset from the row timestamp. The official LeRobot loader assumes video time
equals row time, so the captures cannot be read by simply patching their metadata. Every
capture ships a ``dataloader.py`` that resolves those references, so it is used here as
the source of truth for frame lookup.

``observation.state`` stays 108-dim and ``action`` keeps its original joint order. The
54-dim slicing and the wuji reordering remain in the online transforms
(``openpi.policies.fr3_wuji_policy``), so a future 108-dim model needs no reconversion.

Example:
    uv run examples/fr3_wuji/convert_dataset.py --src-root /home/descfly/datasets
"""

import dataclasses
import importlib.util
import json
import pathlib
import shutil
import sys

from lerobot.common.constants import HF_LEROBOT_HOME
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

SOURCE_LOADER_FILENAME = "dataloader.py"
VECTOR_KEYS = ("observation.state", "action")
CAMERA_KEYS = ("observation.images.cam0", "observation.images.cam1", "observation.images.cam2")
DEFAULT_TASK = (
    "Pick up a tomato truss with the right hand, "
    "then pick a cherry tomato with the left hand and place it in the left basket."
)


@dataclasses.dataclass
class Args:
    # Directory holding the per-episode capture directories.
    src_root: pathlib.Path = pathlib.Path("/home/descfly/datasets")
    # LeRobot repo id of the merged dataset. A relative id keeps norm stats out of the data dir.
    repo_id: str = "fr3_wuji/tomato"
    # Language instruction stored for every episode. Must match what the client sends at inference.
    task: str = DEFAULT_TASK
    # Output root. Defaults to ``HF_LEROBOT_HOME / repo_id``.
    root: pathlib.Path | None = None
    # Video codec. h264 decodes faster than the LeRobot default (libsvtav1).
    vcodec: str = "h264"
    # Convert only the first N captures. Useful for a dry run.
    max_episodes: int | None = None
    image_writer_processes: int = 4
    image_writer_threads: int = 4
    # Delete an existing output directory instead of failing.
    overwrite: bool = False
    # Append remaining captures onto an existing converted dataset.
    resume: bool = False


def _load_source_loader(module_path: pathlib.Path):
    """Import the ``dataloader.py`` shipped alongside the captures."""
    spec = importlib.util.spec_from_file_location("fr3_wuji_source_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source dataloader: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_capture_reader(source_loader):
    """Restrict parquet reads to the columns we actually convert.

    The captures inline a depth payload as a nested struct. Reading it back from a
    multi-row-group file raises ``ArrowNotImplementedError``, and we drop depth anyway.
    """
    columns = ["timestamp", *VECTOR_KEYS, *CAMERA_KEYS]

    class CaptureReader(source_loader.LeRobotEpisodeDataset):
        def _table(self, position: int):
            table = self._tables.get(position)
            if table is None:
                table = pq.read_table(self.episodes[position].parquet_path, columns=columns)
                self._tables[position] = table
            return table

    return CaptureReader


def _episode_sort_key(path: pathlib.Path) -> tuple[int, str]:
    digits = "".join(c for c in path.name if c.isdigit())
    return (int(digits) if digits else 0, path.name)


def _find_captures(src_root: pathlib.Path, max_episodes: int | None) -> list[pathlib.Path]:
    captures = sorted(
        (p for p in src_root.iterdir() if (p / "meta" / "info.json").is_file()),
        key=_episode_sort_key,
    )
    if not captures:
        raise FileNotFoundError(f"No LeRobot captures found under {src_root}")
    return captures[:max_episodes] if max_episodes is not None else captures


def _build_features(info: dict) -> dict:
    features = {}
    for key in VECTOR_KEYS:
        source = info["features"][key]
        features[key] = {
            "dtype": "float32",
            "shape": tuple(source["shape"]),
            "names": source["names"],
        }
    for key in CAMERA_KEYS:
        source = info["features"][key]
        features[key] = {
            "dtype": "video",
            "shape": tuple(source["shape"]),
            "names": ["height", "width", "channel"],
        }
    return features


def _assert_compatible(info: dict, reference: dict, name: str) -> None:
    for key in (*VECTOR_KEYS, *CAMERA_KEYS):
        if key not in info["features"]:
            raise ValueError(f"{name}: missing feature {key}")
        shape = tuple(info["features"][key]["shape"])
        expected = tuple(reference["features"][key]["shape"])
        if shape != expected:
            raise ValueError(f"{name}: {key} has shape {shape}, expected {expected}")
    if info["fps"] != reference["fps"]:
        raise ValueError(f"{name}: fps is {info['fps']}, expected {reference['fps']}")


def _use_codec(vcodec: str) -> None:
    """Route LeRobot's video encoding through a chosen codec."""
    original = lerobot_dataset.encode_video_frames

    def encode(imgs_dir, video_path, fps, **kwargs):
        kwargs.setdefault("vcodec", vcodec)
        return original(imgs_dir, video_path, fps, **kwargs)

    lerobot_dataset.encode_video_frames = encode


def main(args: Args) -> None:
    captures = _find_captures(args.src_root, args.max_episodes)
    reference = json.loads((captures[0] / "meta" / "info.json").read_text())
    for capture in captures:
        _assert_compatible(json.loads((capture / "meta" / "info.json").read_text()), reference, capture.name)

    output_root = args.root if args.root is not None else HF_LEROBOT_HOME / args.repo_id
    if output_root.exists():
        if args.overwrite and args.resume:
            raise ValueError("Pass only one of --overwrite or --resume.")
        if args.overwrite:
            shutil.rmtree(output_root)
        elif not args.resume:
            raise FileExistsError(
                f"{output_root} already exists. Pass --resume to append remaining captures, "
                "or --overwrite to replace the dataset."
            )

    capture_reader = _make_capture_reader(_load_source_loader(captures[0] / SOURCE_LOADER_FILENAME))
    _use_codec(args.vcodec)

    if output_root.exists():
        dataset = lerobot_dataset.LeRobotDataset(repo_id=args.repo_id, root=output_root)
        if args.image_writer_processes or args.image_writer_threads:
            dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
        already_done = dataset.meta.total_episodes
        captures = captures[already_done:]
        print(f"Resuming after {already_done} episodes; {len(captures)} remaining -> {output_root}")
    else:
        dataset = lerobot_dataset.LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=reference["fps"],
            root=output_root,
            robot_type=reference.get("robot_type"),
            features=_build_features(reference),
            use_videos=True,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        print(f"Converting {len(captures)} captures into {output_root}")

    total_frames = dataset.meta.total_frames
    for capture in tqdm.tqdm(captures, desc="episodes", unit="ep"):
        source = capture_reader(
            capture,
            vector_keys=VECTOR_KEYS,
            video_keys=CAMERA_KEYS,
            action_horizon=None,
        )
        try:
            for index in tqdm.tqdm(range(len(source)), desc=capture.name, unit="f", leave=False):
                item = source[index]
                frame = {"task": args.task}
                for key in VECTOR_KEYS:
                    frame[key] = item[key].numpy().astype(np.float32)
                for key in CAMERA_KEYS:
                    # The source loader returns uint8 CHW; LeRobot validates against the HWC shape.
                    frame[key] = item[key].numpy().transpose(1, 2, 0)
                dataset.add_frame(frame)
            dataset.save_episode()
            total_frames += len(source)
        finally:
            source.close()

    print(f"Done: {dataset.meta.total_episodes} episodes, {total_frames} frames -> {output_root}")
    print(f"Set the train config repo_id to {args.repo_id!r} and drop local_dataset_loader.")


if __name__ == "__main__":
    main(tyro.cli(Args))
