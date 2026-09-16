"""Merge compatible LeRobot v2.1 datasets without modifying the sources.

The output episodes and global frame indices are rebuilt sequentially. Parquet
payloads are checked, videos are copied byte-for-byte and the source metadata is
stored under ``provenance/``.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_path(info: dict, kind: str, episode: int, camera: str | None = None) -> str:
    return info[kind].format(
        episode_chunk=episode // info["chunks_size"],
        episode_index=episode,
        video_key=camera,
    )


def compatible_action_names(first: list[str], second: list[str]) -> bool:
    # Some converters add suffixes such as .position/.command without changing
    # the joint order or meaning used by the FR3/Wuji transform.
    return [name.split(".")[0] for name in first] == [name.split(".")[0] for name in second]


def merge(sources: list[Path], output: Path, repo_id: str | None = None) -> None:
    sources = [source.resolve() for source in sources]
    output = output.resolve()
    staging = output.with_name(output.name + ".building")
    if len(sources) < 2:
        raise ValueError("At least two source datasets are required")
    if len(set(sources)) != len(sources):
        raise ValueError("A source dataset was provided more than once")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Output or staging already exists: {output} / {staging}")

    infos = [read_json(source / "meta/info.json") for source in sources]
    task_lists = [read_jsonl(source / "meta/tasks.jsonl") for source in sources]
    unified_tasks = []
    task_identity_to_index = {}
    source_task_maps = []
    for tasks in task_lists:
        task_map = {}
        for task in tasks:
            old_index = task["task_index"]
            identity = json.dumps(
                {key: value for key, value in task.items() if key != "task_index"},
                sort_keys=True,
                ensure_ascii=False,
            )
            if identity not in task_identity_to_index:
                new_index = len(unified_tasks)
                task_identity_to_index[identity] = new_index
                unified_tasks.append({**task, "task_index": new_index})
            task_map[old_index] = task_identity_to_index[identity]
        source_task_maps.append(task_map)

    first_info = infos[0]
    feature_names = list(first_info["features"])
    for source, info in zip(sources, infos, strict=True):
        if info["codebase_version"] != "v2.1":
            raise ValueError(f"{source}: expected LeRobot v2.1")
        for field in ("fps", "robot_type", "chunks_size"):
            if info[field] != first_info[field]:
                raise ValueError(f"{source}: {field} does not match the first dataset")
        if set(info["features"]) != set(feature_names):
            raise ValueError(f"{source}: feature keys do not match the first dataset")
        if info["splits"] != {"train": f"0:{info['total_episodes']}"}:
            raise ValueError(f"{source}: only a contiguous train split is supported")
        for key in feature_names:
            expected = first_info["features"][key]
            actual = info["features"][key]
            if expected["shape"] != actual["shape"] or expected["dtype"] != actual["dtype"]:
                raise ValueError(f"{source}: incompatible feature {key!r}")
            names_match = expected["names"] == actual["names"]
            if key == "action":
                names_match = compatible_action_names(expected["names"], actual["names"])
            if not names_match:
                raise ValueError(f"{source}: incompatible names for feature {key!r}")

    output_info = copy.deepcopy(first_info)
    # Prefer the last source's more descriptive action-name suffixes when only
    # suffixes differ; the joint order has already been checked above.
    output_info["features"]["action"]["names"] = infos[-1]["features"]["action"]["names"]
    cameras = [key for key in feature_names if output_info["features"][key]["dtype"] == "video"]
    columns = [key for key in feature_names if key not in cameras]
    first_data = sources[0] / dataset_path(first_info, "data_path", 0)
    schema = pq.read_schema(first_data).remove_metadata()
    if schema.names != columns:
        raise ValueError("Parquet schema columns do not match meta/info.json")

    staging.mkdir()
    episodes = []
    episode_stats = []
    mapping = []
    source_summaries = []
    global_offset = 0

    for source_number, (source, info) in enumerate(zip(sources, infos, strict=True)):
        source_episodes = read_jsonl(source / "meta/episodes.jsonl")
        stats_by_episode = {
            item["episode_index"]: item["stats"] for item in read_jsonl(source / "meta/episodes_stats.jsonl")
        }
        expected_indices = list(range(info["total_episodes"]))
        if [item["episode_index"] for item in source_episodes] != expected_indices:
            raise ValueError(f"{source}: episode indices are not contiguous")
        shutil.copytree(source / "meta", staging / f"provenance/source_{source_number}/meta")

        source_offset = 0
        for episode in source_episodes:
            old_index = episode["episode_index"]
            new_index = len(episodes)
            length = episode["length"]
            source_data = source / dataset_path(info, "data_path", old_index)
            source_hash = digest(source_data)
            original = pq.read_table(source_data, columns=columns).cast(schema)
            if len(original) != length:
                raise ValueError(f"{source}: episode {old_index} length mismatch")
            if not np.array_equal(original["episode_index"].to_numpy(), np.full(length, old_index)):
                raise ValueError(f"{source}: invalid episode_index in episode {old_index}")
            if not np.array_equal(original["index"].to_numpy(), np.arange(source_offset, source_offset + length)):
                raise ValueError(f"{source}: invalid global index in episode {old_index}")
            if not np.array_equal(original["frame_index"].to_numpy(), np.arange(length)):
                raise ValueError(f"{source}: invalid frame_index in episode {old_index}")
            original_task_indices = original["task_index"].to_numpy()
            try:
                remapped_task_indices = np.asarray(
                    [source_task_maps[source_number][int(value)] for value in original_task_indices]
                )
            except KeyError as error:
                raise ValueError(f"{source}: undefined task_index {error.args[0]}") from error

            table = original
            replacements = (
                ("episode_index", np.full(length, new_index)),
                ("index", np.arange(global_offset, global_offset + length)),
                ("task_index", remapped_task_indices),
            )
            for key, values in replacements:
                field = schema.field(key)
                table = table.set_column(
                    schema.get_field_index(key),
                    field,
                    pa.array(values, type=field.type),
                )
            target_data = staging / dataset_path(output_info, "data_path", new_index)
            target_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target_data, compression="snappy")
            actual = pq.read_table(target_data)
            if not actual.equals(table):
                raise ValueError(f"Parquet roundtrip failed for output episode {new_index}")

            video_hashes = {}
            for camera in cameras:
                source_video = source / dataset_path(info, "video_path", old_index, camera)
                target_video = staging / dataset_path(output_info, "video_path", new_index, camera)
                target_video.parent.mkdir(parents=True, exist_ok=True)
                video_hash = digest(source_video)
                shutil.copy2(source_video, target_video)
                if digest(target_video) != video_hash:
                    raise ValueError(f"Video copy verification failed: {target_video}")
                video_hashes[camera] = video_hash
                with av.open(str(target_video)) as container:
                    stream = container.streams.video[0]
                    if stream.frames != length or float(stream.average_rate) != output_info["fps"]:
                        raise ValueError(f"Video timing mismatch: {target_video}")
                    if [stream.height, stream.width, 3] != output_info["features"][camera]["shape"]:
                        raise ValueError(f"Video shape mismatch: {target_video}")

            new_stats = {key: copy.deepcopy(stats_by_episode[old_index][key]) for key in feature_names}
            for key in ("episode_index", "index", "task_index"):
                values = actual[key].to_numpy().astype(np.float64)
                new_stats[key] = {
                    "min": [float(values.min())],
                    "max": [float(values.max())],
                    "mean": [float(values.mean())],
                    "std": [float(values.std())],
                    "count": [length],
                }
            episodes.append({**episode, "episode_index": new_index})
            episode_stats.append({"episode_index": new_index, "stats": new_stats})
            mapping.append(
                {
                    "episode_index": new_index,
                    "source_dataset": str(source),
                    "source_episode_index": old_index,
                    "length": length,
                    "source_parquet_sha256": source_hash,
                    "video_sha256": video_hashes,
                }
            )
            global_offset += length
            source_offset += length

        if source_offset != info["total_frames"]:
            raise ValueError(f"{source}: total frame count mismatch")
        source_summaries.append(
            {
                "root": str(source),
                "episodes": len(source_episodes),
                "frames": source_offset,
            }
        )
        print(f"Merged {source}: {len(source_episodes)} episodes, {source_offset} frames", flush=True)

    total_episodes = len(episodes)
    output_info.update(
        total_episodes=total_episodes,
        total_frames=global_offset,
        total_videos=total_episodes * len(cameras),
        total_chunks=math.ceil(total_episodes / output_info["chunks_size"]),
        splits={"train": f"0:{total_episodes}"},
    )
    write_json(staging / "meta/info.json", output_info)
    write_jsonl(staging / "meta/tasks.jsonl", unified_tasks)
    write_jsonl(staging / "meta/episodes.jsonl", episodes)
    write_jsonl(staging / "meta/episodes_stats.jsonl", episode_stats)
    write_json(
        staging / "provenance/merge.json",
        {
            "repo_id": repo_id or f"fr3_wuji/{output.name}",
            "sources": source_summaries,
            "episodes": total_episodes,
            "frames": global_offset,
            "videos": total_episodes * len(cameras),
            "mapping": mapping,
            "sampling": "Physical merge only; no normalization statistics were computed.",
            "checks": [
                "Compatible LeRobot v2.1 metadata and feature schemas",
                "Sequential episode, frame and global indices",
                "Parquet roundtrip verification",
                "Video byte hashes and stream metadata verification",
            ],
        },
    )
    shutil.copy2(__file__, staging / "provenance/merge_script.py")
    staging.rename(output)
    print(f"READY: {output} ({total_episodes} episodes, {global_offset} frames)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    merge(args.sources, args.output, args.repo_id)


if __name__ == "__main__":
    main()
