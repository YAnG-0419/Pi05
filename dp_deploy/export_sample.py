"""Export a two-frame raw NPZ window from an existing converted DP Zarr (read-only)."""

import argparse
from pathlib import Path
import numpy as np
from .contract import RGB_KEYS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zarr")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("reports/sample.npz"))
    a = p.parse_args()
    import zarr

    z = zarr.open(a.zarr, mode="r")
    i = a.index
    if i < 0 or i + 1 >= len(z["data/joint_pos"]):
        raise ValueError("Need two valid frames")
    ends = z["meta/episode_ends"][:]
    if np.any(ends == i + 1):
        raise ValueError("Window crosses episode boundary")
    if (
        z.attrs["group_order"] != ["left_arm", "right_arm", "left_hand", "right_hand"]
        or z.attrs["fps"] != 30
        or z.attrs["depth_max_m"] != 2
    ):
        raise ValueError("Dataset contract differs from deployed checkpoint")
    arrays = {
        k: z["data/" + k][i : i + 2] for k in (*RGB_KEYS, "head_depth", "joint_pos")
    }
    arrays["head_depth"] = arrays["head_depth"][..., 0]
    arrays["timestamps"] = (
        z["provenance/timestamp_ns"][i : i + 2].astype(np.float64) / 1e9
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **arrays)
    print(a.output)


if __name__ == "__main__":
    main()
