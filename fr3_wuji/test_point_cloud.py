import json

import numpy as np
import pytest

from point_cloud import crop
from point_cloud import farthest_point_sample
from point_cloud import load_crop
from point_cloud import reconstruct
from point_cloud import voxel_downsample


def test_reconstruct_crop_and_sample(tmp_path):
    config = tmp_path / "crop.json"
    config.write_text(json.dumps({
        "coordinate_frame": "cam0_depth_optical_frame",
        "crop_min": [-1, -1, 0.5],
        "crop_max": [1, 1, 2],
        "planes": [],
    }))
    lower, upper, planes = load_crop(config)
    depth = np.full((40, 40), 1000, dtype=np.uint16)
    points = reconstruct(depth, [[100, 0, 20], [0, 100, 20], [0, 0, 1]])
    points = crop(points, lower, upper, planes)
    assert points.shape == (1600, 3)
    assert voxel_downsample(points).shape[1] == 3
    sampled = farthest_point_sample(points, 1024)
    assert sampled.shape == (1024, 3)
    assert np.isfinite(sampled).all()


def test_too_few_points_rejected():
    with pytest.raises(ValueError, match="need 4"):
        farthest_point_sample(np.zeros((3, 3), dtype=np.float32), 4)


def test_float_depth_is_already_metres():
    depth = np.ones((2, 2), dtype=np.float32)
    points = reconstruct(depth, np.eye(3))
    np.testing.assert_array_equal(points[:, 2], np.ones(4))
