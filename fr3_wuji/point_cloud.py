"""Training-identical RGB-D point-cloud preprocessing using NumPy only."""

import json
from pathlib import Path

import numpy as np


def load_crop(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("coordinate_frame") != "cam0_depth_optical_frame":
        raise ValueError("Crop config must use cam0_depth_optical_frame")
    lower = np.asarray(config.get("crop_min"), dtype=np.float32)
    upper = np.asarray(config.get("crop_max"), dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("Invalid crop bounds")
    planes = []
    for item in config.get("planes", []):
        normal = np.asarray(item.get("normal"), dtype=np.float64)
        length = np.linalg.norm(normal)
        offset, margin = float(item.get("offset", 0)), float(item.get("margin", 0))
        if normal.shape != (3,) or not np.isfinite(normal).all() or length < 1e-12:
            raise ValueError("Invalid crop plane normal")
        if not np.isfinite([offset, margin]).all() or margin < 0:
            raise ValueError("Invalid crop plane offset/margin")
        planes.append((normal / length, offset / length, margin))
    return lower, upper, planes


def reconstruct(depth, intrinsics, *, depth_scale=0.001, min_depth=0.001, max_depth=10.0):
    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("Expected depth [H,W] and intrinsics [3,3]")
    if not np.isfinite(intrinsics).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    z = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        z *= depth_scale
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    rows, cols = np.nonzero(valid)
    z = z[valid]
    x = (cols - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def crop(points, lower, upper, planes):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Point cloud must be finite [N,3]")
    keep = np.all((points >= lower) & (points <= upper), axis=1)
    for normal, offset, margin in planes:
        keep &= points @ normal + offset >= margin
    return points[keep]


def voxel_downsample(points, size=0.005):
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Voxel size must be positive")
    keys = np.floor(points / size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def farthest_point_sample(points, count=1024):
    if len(points) < count:
        raise ValueError(f"Only {len(points)} points remain; need {count}")
    selected = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf, dtype=np.float32)
    farthest = int(np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1)))
    for index in range(count):
        selected[index] = farthest
        delta = points - points[farthest]
        distance = np.minimum(distance, np.einsum("ij,ij->i", delta, delta))
        farthest = int(np.argmax(distance))
    return points[selected].astype(np.float32, copy=False)


def prepare(depth, intrinsics, crop_config):
    lower, upper, planes = load_crop(crop_config)
    points = crop(reconstruct(depth, intrinsics), lower, upper, planes)
    if len(points) < 1024:
        raise ValueError(f"Crop contains only {len(points)} valid points")
    candidates = voxel_downsample(points)
    return farthest_point_sample(candidates if len(candidates) >= 1024 else points)
