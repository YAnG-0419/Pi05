"""Offline single-observation inference; never connects to a robot.

Run from the repository root:
    uv run python -m examples.fr3_wuji.replay_check
"""

import dataclasses
import pathlib
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi.policies import fr3_wuji_policy
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


@dataclasses.dataclass
class Args:
    checkpoint: pathlib.Path = pathlib.Path("checkpoints/pi05_fr3_wuji/tomato_lora_65ep/19999")
    repo_id: str = "fr3_wuji/tomato"
    # Converted dataset episode index (0..64), not the original capture filename.
    episode: int = 0
    # Local frame within the episode; -1 selects the middle.
    frame: int = -1
    seed: int = 0
    output_dir: pathlib.Path = pathlib.Path("replay_checks")


GROUPS = (("left_arm", 0, 7), ("left_hand", 7, 27), ("right_arm", 27, 34), ("right_hand", 34, 54))


def main(args: Args) -> None:
    cfg = config.get_config("pi05_fr3_wuji")
    horizon = cfg.model.action_horizon
    if not (args.checkpoint / "params").is_dir():
        raise ValueError(f"Missing checkpoint params: {args.checkpoint}")
    ds = LeRobotDataset(args.repo_id)
    if not 0 <= args.episode < ds.meta.total_episodes:
        raise ValueError(f"episode must be in [0, {ds.meta.total_episodes - 1}]")
    length = ds.meta.episodes[args.episode]["length"]
    frame = length // 2 if args.frame == -1 else args.frame
    if not 0 <= frame <= length - horizon:
        raise ValueError(f"frame must be in [0, {length - horizon}] to fit a full action chunk")
    index = sum(ds.meta.episodes[e]["length"] for e in range(args.episode)) + frame
    item = ds[index]
    obs = {
        "observation/state": item["observation.state"].numpy(),
        "observation/image": item["observation.images.cam0"].numpy(),
        "observation/left_wrist_image": item["observation.images.cam1"].numpy(),
        "observation/right_wrist_image": item["observation.images.cam2"].numpy(),
        "prompt": ds.meta.tasks[int(item["task_index"].item())],
    }
    raw_actions = np.stack([ds[i]["action"].numpy() for i in range(index, index + horizon)])
    transformed = fr3_wuji_policy.Fr3WujiInputs(cfg.model.model_type)({**obs, "actions": raw_actions})
    state, target = transformed["state"], transformed["actions"]
    print(f"Episode index={args.episode}, local frame={frame}/{length}, global row={index}", flush=True)
    print(f"Prompt: {obs['prompt']}", flush=True)
    print(f"Loading {args.checkpoint} ...", flush=True)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    noise = np.random.default_rng(args.seed).standard_normal((horizon, 54)).astype(np.float32)
    start = time.monotonic()
    pred = policy.infer(obs, noise=noise)["actions"]
    print(f"Inference including first compilation: {time.monotonic() - start:.2f}s")
    if pred.shape != (horizon, 54) or not np.isfinite(pred).all():
        raise ValueError(f"Invalid output: shape={pred.shape}, finite={np.isfinite(pred).all()}")

    print("Output shape: (50, 54); all values finite. Units are the dataset's joint-position units.")
    print("group        MAE vs demo   first jump   demo first jump   max chunk step   demo chunk step")
    for name, a, b in GROUPS:
        p, t, s = pred[:, a:b], target[:, a:b], state[a:b]
        print(
            f"{name:12s} {np.abs(p - t).mean():11.5f} {np.abs(p[0] - s).max():12.5f} "
            f"{np.abs(t[0] - s).max():17.5f} {np.abs(np.diff(p, axis=0)).max():16.5f} "
            f"{np.abs(np.diff(t, axis=0)).max():17.5f}"
        )
        j = int(np.argmax(np.abs(p[0] - s)))
        print(
            f"  Largest first jump: {name}[{j}] (global dim {a + j}), "
            f"state={s[j]:.5f}, prediction={p[0, j]:.5f}, demo={t[0, j]:.5f}"
        )

    # Only hands are absolute in the training action statistics; arm stats are deltas.
    stats = normalize.load(args.checkpoint / "assets" / args.repo_id)["actions"]
    if stats.q01 is not None and stats.q99 is not None:
        for name, a, b in (GROUPS[1], GROUPS[3]):
            outside = (pred[:, a:b] < stats.q01[a:b]) | (pred[:, a:b] > stats.q99[a:b])
            print(f"{name}: {outside.sum()}/{outside.size} values outside training action q01..q99")
    print("Quantile outliers are NOT hardware limit violations. Hardware limits are not checked.")
    print("Finger opening/closing direction requires the hand joint convention; no automatic verdict here.")

    # A separate timestamped directory preserves previous checks.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import tempfile

    out = pathlib.Path(tempfile.mkdtemp(prefix=f"ep{args.episode}_frame{frame}_", dir=args.output_dir))
    np.savez(
        out / "actions.npz",
        state=state,
        prediction=pred,
        demonstration=target,
        episode=args.episode,
        frame=frame,
        fps=ds.meta.fps,
        seed=args.seed,
        checkpoint=str(args.checkpoint),
    )
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    times = np.arange(horizon) / ds.meta.fps
    for name, a, b in GROUPS:
        fig, axes = plt.subplots((b - a + 3) // 4, 4, figsize=(16, 2.5 * ((b - a + 3) // 4)), squeeze=False)
        for j, ax in enumerate(axes.flat):
            if a + j >= b:
                ax.set_visible(False)
                continue
            ax.plot(times, pred[:, a + j], label="prediction")
            ax.plot(times, target[:, a + j], label="demonstration", linestyle="--")
            ax.axhline(state[a + j], color="gray", linestyle=":", label="current state")
            ax.set_title(f"{name}[{j}] / dim {a + j}")
            ax.set_xlabel("seconds")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)
    print(f"Saved action arrays and four joint-curve plots to: {out.resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
