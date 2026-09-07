# FR3 + Wuji Hand 微调待办

在官方 openpi 工作区（本仓库）微调 `pi05_base`。参考 [wuji-openpi](https://github.com/wuji-technology/wuji-openpi) 的 54 维约定，**第一阶段在线 transform，不改磁盘数据**。用 `data/dataset/episode24`（1 条、2377 帧）跑通后，把 `/home/descfly/datasets` 的 65 条整理成标准训练集。

任务指令（训练与部署必须**逐字一致**）：

```text
Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.
```

相关参考：

- 模型输入约定：`src/openpi/models/model.py`（3 路 RGB，无深度）
- 官方适配模板：`src/openpi/policies/libero_policy.py`、`LeRobotLiberoDataConfig`
- wuji 参考：`/home/descfly/lpy/wuji-openpi/src/openpi/policies/wuji_policy.py`、`LeRobotWujiDataConfig`、`PartialCheckpointWeightLoader`
- wuji 仓库参考：`/home/descfly/lpy/wuji-openpi`

---

## 已确认的数据事实

| 项目 | 当前 `episode24` | 训练时要对齐的空间 |
| --- | --- | --- |
| 磁盘格式 | LeRobot v2.0 | 保持不变（阶段 1） |
| 相机 | `cam0` 头 / `cam1` 左腕 / `cam2` 右腕 | 映射到 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` |
| 深度 `observation.depths.cam0` | 有 | **不使用** |
| `observation.engaged` | 有 | **不使用** |
| `observation.state` | 108 维（位置 + 速度） | **只要位置 54 维，并重排** |
| `action` | 54 维，顺序 `[左臂7, 右臂7, 左手20, 右手20]` | **重排为** `[左臂7, 左手20, 右臂7, 右手20]` |
| `task` | `teleoperation` | 能跑；真训前改成自然语言指令 |
| fps | 元数据 30（`cam0` 视频 20） | 先按 30 跑通，有对不齐再查 |
| LeRobot 版本 | 数据集 `v2.0` | 已安装 lerobot 是 `v2.1`，**向后兼容**，加载仅告警 |

> **已实测（用当前 `.venv` 加载 `episode24`）：** `fps=30`、`episodes=1`、
> `frames=2377`、`action.shape=(54,)`、`state.shape=(108,)`、`tasks={0: 'teleoperation'}`，
> 能正常加载。该数据虽然标记为 v2.0，但缺少当前 LeRobot API 必需的 `chunks_size`、
> `data_path` 等字段，因此阶段 1 通过数据目录内自带的 `dataloader.py` 在线读取，
> 不修改磁盘数据；升级成标准 v2.1 后可移除此兼容配置。

### state 108 → 54（丢掉全部速度）

当前顺序：

```text
[左臂pos 0:7] [左臂vel 7:14] [右臂pos 14:21] [右臂vel 21:28]
[左手pos 28:48] [左手vel 48:68] [右手pos 68:88] [右手vel 88:108]
```

目标（与 wuji / delta mask 一致）：

```text
[左臂pos 0:7] [左手pos 28:48] [右臂pos 14:21] [右手pos 68:88]
```

```python
state_54 = np.concatenate(
    [state[..., 0:7], state[..., 28:48], state[..., 14:21], state[..., 68:88]],
    axis=-1,
)
```

### action 重排（维数已是 54）

```python
action_wuji = np.concatenate(
    [action[..., 0:7], action[..., 14:34], action[..., 7:14], action[..., 34:54]],
    axis=-1,
)
```

`extra_delta_transform` 的 mask 是 `make_bool_mask(7, -20, 7, -20)`：臂做 delta，手保持绝对位置。**必须先重排再做 delta**，否则会把左手前 7 维当成右臂。

---

## 阶段 0：环境与权重

- [x] 本仓库 `uv sync` 可用，能 `uv run python -c "import openpi; import lerobot"`
- [x] 确认 `pi05_base` 权重已在本地缓存，或训练时能拉 `gs://openpi-assets/checkpoints/pi05_base/params`
- [x] GPU / 显存心里有数：RTX 5090 32GB，全量微调放不下（约需 50GiB），改走 LoRA
- [x] 冒烟关掉 wandb（`wandb_enabled=False`）

---

## 阶段 1：在官方 openpi 里做适配（在线 transform）

目标：磁盘仍指向 `data/dataset/episode24`，只加代码。

### 1.1 搬 `PartialCheckpointWeightLoader`

- [x] 从 `wuji-openpi/src/openpi/training/weight_loaders.py` 把 `PartialCheckpointWeightLoader` 拷到本仓库同名文件
- [x] 默认跳过形状对不上的 `action_in_proj` / `action_out_proj` / `state_proj`（32 维 → 54 维必须跳过，这些层随机初始化）
- [x] 不要用官方的 `CheckpointWeightLoader` 硬灌 54 维模型，会形状对不上

### 1.2 新增 policy transform

- [x] 新增 `src/openpi/policies/fr3_wuji_policy.py`（以 `libero_policy.py` + wuji 的 `WujiInputs` 为模板）
- [x] `Fr3WujiInputs`（训练和推理都走这里）：
  - [x] 解析三路图：`observation/image`=cam0，`left_wrist_image`=cam1，`right_wrist_image`=cam2 → uint8 HWC
  - [x] 填 `image` / `image_mask` 三个固定键（都 `True`）
  - [x] `state`：108 则按上面切片；已是 54 则原样（给阶段 3 离线数据留口）
  - [x] `actions`：若存在则重排到 wuji 顺序（注意可能是 `(horizon, 54)`）
  - [x] 透传 `prompt`
- [x] `Fr3WujiOutputs`：推理时 `actions[:, :54]`，顺序已是 `[左臂, 左手, 右臂, 右手]`
- [x] `make_fr3_wuji_example()`：54 维 state + 三张图，方便单测 / 空跑 policy

### 1.3 新增 DataConfig + TrainConfig

文件：`src/openpi/training/config.py`

- [x] `import openpi.policies.fr3_wuji_policy as fr3_wuji_policy`
- [x] 新增 `LeRobotFr3WujiDataConfig`（抄 `LeRobotLiberoDataConfig` / wuji 的 `LeRobotWujiDataConfig`）
  - [x] **repack**（只作用于数据集，不作用于推理）：

    ```python
    {
        "observation/image": "observation.images.cam0",
        "observation/left_wrist_image": "observation.images.cam1",
        "observation/right_wrist_image": "observation.images.cam2",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    }
    ```

  - [x] **不要**把 depth / engaged 写进 repack
  - [x] `action_sequence_keys=("action",)`（磁盘列名是 `action`；`DataConfig` 默认是 `actions`，不改会读不到 chunk）
  - [x] `data_transforms`：`Fr3WujiInputs` + `Fr3WujiOutputs`
  - [x] `extra_delta_transform=True`：`make_bool_mask(7, -20, 7, -20)`，接在 Inputs **之后**
  - [x] `prompt_from_task=True`
- [x] 新增 `TrainConfig(name="pi05_fr3_wuji")`：
  - [x] `model=Pi0Config(pi05=True, action_dim=54, action_horizon=50, max_token_len=256)`
  - [x] `repo_id` 先用绝对路径：`/home/descfly/lpy/openpi/data/dataset/episode24`
    - 实测：openpi 只调 `LeRobotDatasetMetadata(repo_id)`（不传 `root`），内部按
      `HF_LEROBOT_HOME / repo_id` 定位；绝对路径会被 `pathlib` 直接采用，**能加载**。
    - 备选（避免 norm stats 落进数据集目录）：`repo_id="fr3_wuji/episode24"` 之类的相对 id，
      并把数据集软链或放到 `~/.cache/huggingface/lerobot/fr3_wuji/episode24`。
  - [x] `weight_loader=PartialCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
  - [x] 冒烟建议：`batch_size=2`，`num_train_steps=20`，`save_interval=10`，`wandb_enabled=False`，`overwrite=True`
  - [ ] 真训再改回更大 batch / `30_000` steps（按显存调；wuji 默认 64 多半要减小）

### 1.4 冒烟前自检（先于训）

写一个小脚本或 notebook，用同一套 transform 打一条样本：

- [x] 本地兼容 loader 能打开 `episode24`，帧数 2377
- [x] 三路图能解码，形状为 cam0 `400x640x3`，cam1/2 `480x640x3`
- [x] transform 后 `state.shape[-1] == 54`
- [x] transform 后 `actions.shape == (50, 54)`（或 `action_horizon`）
- [x] 抽一帧核对：左臂 7 维对得上原始 `action[0:7]`，左手对得上原始 `action[14:34]`
- [x] 深度列存在但不进入 batch

---

## 阶段 2：用 1 条 episode 跑通训练

norm 必须在 **transform 之后** 的 54 维空间上算（`compute_norm_stats.py` 已经会跑 `repack + data_transforms`）。

```bash
# 在仓库根目录
uv run scripts/compute_norm_stats.py --config-name pi05_fr3_wuji
```

> **实测确认的保存路径（重要）：** `compute_norm_stats.py` 的保存路径是
> `config.assets_dirs / data_config.repo_id`，即 `./assets/pi05_fr3_wuji/<repo_id>`。
> 因为我们的 `repo_id` 是**绝对路径**，`pathlib` 拼接会丢弃左侧，最终 norm stats
> 会写进**数据集目录本身**：`/home/descfly/lpy/openpi/data/dataset/episode24/`。
> 训练时加载路径是 `assets_dirs / asset_id`，而 `asset_id` 默认就等于 `repo_id`，
> 所以 save/load 会指向同一处、**能对上**（只是文件落在了数据集目录，属正常）。
>
> - [ ] **不要**给这个 config 设自定义 `asset_id`：保存按 `repo_id`、加载按 `asset_id`，
>   两者不一致会导致训练时找不到 norm stats。
> - [ ] 若不想污染数据集目录，可改用「相对 id + 放到 `~/.cache/huggingface/lerobot/<id>`」
>   的方式（见阶段 1.3 备注），而不是绝对路径。

- [x] 命令成功，`state` / `actions` 统计维数是 54，不是 108（norm_stats 落在 `data/dataset/episode24/norm_stats.json`）
- [x] 日志里没有缺视频、缺 `action` 列、shape mismatch

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_fr3_wuji --exp-name=smoke_episode24 --overwrite
```

- [x] 权重 partial load 日志：跳过 `action_in_proj` / `action_out_proj`，其余层加载成功
- [x] 前几 step loss 是有限数字（step 0 `loss=1.5940`，非 NaN）
- [x] checkpoint 写到 `checkpoints/pi05_fr3_wuji/smoke_episode24/19/`
- [x] 能起服务（用 `create_trained_policy` 打一帧，返回 `(50, 54)`）：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_fr3_wuji \
  --policy.dir=checkpoints/pi05_fr3_wuji/smoke_episode24/<step>
```

- [x] 用 `make_fr3_wuji_example()` 打一帧，返回 `(50, 54)`

**阶段 2 通过标准：** 数据能进模型、能存 ckpt、能 serve。1 条数据过拟合很正常，不看任务成功率。

> **重要变更（阶段 2 实测）：**
> 1. **单卡 32GB（RTX 5090）跑不动 pi05 全量微调**（初始化约需 50GiB）。已把
>    `pi05_fr3_wuji` 改成 **LoRA 微调**（`gemma_2b_lora` + `gemma_300m_lora`，`ema_decay=None`），
>    可放进 32GB。阶段 3 正式训练继续沿用 LoRA，除非换更大/多卡显存。
> 2. **openpi 的 `TorchDataLoader` 用 `spawn` 多进程**，动态 import 的数据集 loader
>    无法被 worker 反序列化。已在 `data_loader.py` 加 `LocalLeRobotDataset`
>    包装类（`__getstate__/__setstate__` 在 worker 里重建内部对象），spawn 下可 pickle。
> 3. norm stats / checkpoint 的 `assets/` 均因绝对 `repo_id` 落到数据集目录，save/load
>    路径一致、能对上（checkpoint 非自包含，属预期）。

---

## 阶段 3：65 条数据整理

### 3.0 源数据实测（`/home/descfly/datasets`，65 条）

65 条 schema **完全一致**：`state` 108 维、`action` 54 维、`fps=30`、三路相机、v2.0、
额外列 `observation.depths.cam0` + `observation.engaged`。合计 **169,282 帧（约 1.57 小时）**，18GB。

两个决定转换方案的坑：

1. **每路相机有自己的时间基准和偏移。** parquet 里视频引用是
   `{path, timestamp}`，且 `frame_index = round(timestamp * 该相机 fps)`。实测
   cam0 从 1.05s 起、步长 0.05s（**20fps**），cam1/cam2 从 1.0s 起、步长 0.033s（30fps），
   而行 `timestamp` 从 0 起。官方 LeRobot 假设「视频时间 == 行时间」，
   **只补元数据是不够的**，直接交给官方 loader 会取错帧（cam0 差约 31 帧）。
2. **inline depth 读不动。** 源 `dataloader.py` 的 `pq.read_table` 不限定列，
   把嵌套 struct 的 depth 也读进来；行数超过单个 row group 时报
   `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs`。
   episode24（2377 行）够小没触发，episode18（4640 行）必然触发。

### 3.1 离线转换成一个标准 LeRobot 数据集

脚本：`examples/fr3_wuji/convert_dataset.py`，用 `LeRobotDataset.create` + `add_frame` +
`save_episode` 官方 API 产出，格式正确性由 LeRobot 自己保证。

- [x] 用数据集自带 `dataloader.py` 作为取帧真源（它已正确处理每相机偏移与 cam0 的 20fps）
- [x] 子类覆盖 `_table`，只读 `timestamp` + 向量列 + 视频列，绕开 inline depth 的 pyarrow 限制
- [x] 65 条合并成一个库并重编号（源数据每条 `episode_index` 都是 0）
- [x] `tasks.jsonl` 写真实英文指令
- [x] **磁盘保留 `state` 108 维 + `action` 原顺序**，54 维切片和重排仍留在 `Fr3WujiInputs`
      —— 不动已验证的公式，且以后训 108 维不用重转数据
- [x] 丢掉 depth / engaged
- [x] 视频用 h264 重编码（解码比 LeRobot 默认的 libsvtav1 快；`g=2` 使随机 seek 很便宜）

> **为什么不存逐帧图片：** LeRobot 的 `image` dtype 不是松散 JPEG 文件，而是把 PNG 字节
> **嵌进 parquet**（`_save_episode_table` 里 `embed_images`），65 条约 150GB 且编码 50 万张图很慢。
> 而标准视频路径 `g=2` 本来就 seek 便宜，体积只有约 2.9GB，全面更优。

### 3.2 转换后校验

脚本：`examples/fr3_wuji/verify_dataset.py`（`uv run python -m examples.fr3_wuji.verify_dataset`）。

向量必须逐维完全相等；图像因有损重编码不能比特级比较，改判**帧对齐**：转换帧必须比任何
邻近源帧更接近自己的源帧。两个必须的门控，否则测试没有判别力：

- 按**每个相机各自**的运动挑帧。静止片段上相邻帧差（约 0.1）远小于编码噪声（约 3～5），
  argmin 会随机落到 ±1，产生假阳性。
- 排除与自身几乎无差别的邻居。cam0 是 20fps，相邻行会映射到**同一源帧**，构成平局。

- [x] 2 条 dry run：向量 max diff **0**，23 次对齐检查最小余量 **3.63**（高运动帧上偏移 0 以 3.18 vs 10.41 完胜）

### 3.3 切到新数据集

- [x] `repo_id` 改成相对 id `fr3_wuji/tomato`，norm stats 落回 `./assets/pi05_fr3_wuji/`，不再污染数据集目录
- [x] 去掉 `local_dataset_loader` / `local_dataset_video_keys`
- [ ] **重新** `compute_norm_stats.py`（不能复用 episode24 的统计）
- [ ] 冒烟确认新数据路径能训（`--num-train-steps 20 --batch-size 2 --overwrite`）
- [ ] 移除 `data_loader.py` 里已无用的 `LocalLeRobotDataset` 兼容层
- [ ] 正式训：`num_train_steps=20_000`，按显存调 `batch_size`，需要时打开 wandb

### 3.4 正式训练经验值

- [ ] batch 按显存：LoRA + 32GB 先试 8，OOM 再降
- [ ] steps：先 10k～20k，看 train loss 和过拟合
- [ ] `action_horizon=50` 与控制 30Hz 对齐（推理约 0.6 Hz 换 chunk）；不要和 wuji 训练 config 里的 100 混用，除非部署也改
- [ ] 仍用 `PartialCheckpointWeightLoader` + `pi05_base`

---

## 阶段 4：部署（训练验证后再排）

- [ ] 参考 `wuji-openpi/examples/wuji`，话题改成 FR3 + Wuji Hand 2（不是 `/tianji_arm/...`）
- [ ] 发给 policy 的图像键与 `Fr3WujiInputs` 一致（repack 不在推理路径）
- [ ] 真机 state 若仍是 108 维，走 Inputs 切片；若客户端已拼 54 维，顺序必须是 `[左臂, 左手, 右臂, 右手]`
- [ ] 下发命令按同一顺序拆成左臂 / 左手 / 右臂 / 右手
- [ ] `serve_policy` 的 `--policy.config` 与训练 config 同名，norm stats 从 checkpoint 的 `assets` 读

---

## 不要做的事

- 不要把 depth 接进 SigLIP（`pi05` 只有 3 路 RGB）
- 不要在官方仓库用 `CheckpointWeightLoader` 直接灌 54 维
- 不要阶段 1 就复制/转码整份视频
- 不要混用「108 维算的 norm」和「54 维训的模型」
- 不要忘记 action 重排（只切速度不够）
- 不要用 1 条 episode 的 ckpt 评估任务（只能验证管线）

---

## 建议实施顺序

1. `PartialCheckpointWeightLoader` + `fr3_wuji_policy.py` + config  
2. 单样本 shape / 重排检查  
3. `compute_norm_stats` + 20 step 冒烟 + `serve_policy`  
4. 勾选「阶段 2 通过」  
5. 整理 70 条（先合并 + 改 task；可选再离线 54 维）  
6. 重算 norm、正式微调  
7. 改部署话题上真机  

阶段 1～2 的代码可以在本仓库直接开做；阶段 3 脚本等冒烟过后再写，避免两套公式先分叉。
