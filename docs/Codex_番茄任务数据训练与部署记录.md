# FR3/Wuji 番茄任务：数据、训练与部署记录

## 1. 目标

基于 OpenPI 的 `pi05_base`，使用 FR3 双臂和 Wuji 灵巧手的番茄任务数据进行 LoRA 微调。任务指令为：

> Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.

模型输入使用三路 RGB 图像、机器人状态和任务文本，输出未来 50 步、54 维的绝对关节位置指令。

## 2. 数据位置和结构

旧数据的原始采集目录：

```text
/home/descfly/datasets
```

旧数据转换后的 LeRobot v2.1 数据集：

```text
/home/descfly/.cache/huggingface/lerobot/fr3_wuji/tomato
```

旧数据规模：65 条 episode、169282 帧。

新数据最初存放在：

```text
/home/descfly/lpy/openpi/data/tomato/tomato_standard_quality
```

新数据本身已经是 LeRobot v2.1 格式，包含：

- 61 条 episode、116398 帧，30 FPS
- `observation.state`：108 维
- `action`：54 维
- 三路 H.264 RGB 视频：`cam0`、`cam1`、`cam2`
- 与旧数据相同的英文任务指令

检查发现原 `episode_000058` 有 45 帧的右臂 `engaged=0`。按照决定，完整移除了该 episode，并重新编号后续 episode、更新 parquet 内的 episode/global index、元数据和统计。

清理后的新数据集：

```text
/home/descfly/lpy/openpi/data/tomato/tomato_standard_quality_60ep
```

规模：60 条 episode、116154 帧、180 个视频。原始 61 条数据仍完整保留。

## 3. 联合数据集

为了用旧 65 条和新 60 条训练同一个模型，生成了独立的联合数据集：

```text
/home/descfly/lpy/openpi/data/tomato/tomato_joint_125ep
```

联合数据规模：

- 125 条 episode
- 285436 帧
- 375 个视频
- 30 FPS
- 一个相同任务

合并时保留了两份源数据，视频内容未重新编码。联合数据的 episode 编号和全局 frame index 已连续重排，并完成 parquet、视频头、任务文本和加载器检查。

LeRobot 缓存软链接：

```text
~/.cache/huggingface/lerobot/fr3_wuji/tomato_joint_125ep
  -> /home/descfly/lpy/openpi/data/tomato/tomato_joint_125ep
```

## 4. 数据变换约定

当前训练配置使用 `LeRobotFr3WujiDataConfig`。

磁盘上的 108 维 state 经过在线切片和重排后，模型实际使用 54 维：

```text
[左臂 7, 左手 20, 右臂 7, 右手 20]
```

磁盘上的 action 原始顺序也会在线重排为相同的 54 维顺序。模型 action horizon 为 50。

机械臂 14 个维度在归一化前转换为相对当前 state 的 delta；手部维度保持绝对关节位置。模型输出经过反归一化后，机械臂 delta 会恢复为绝对关节位置，因此部署端不要再次累加当前状态。

## 5. Norm stats

Norm stats 必须与实际参与训练的数据及其在线 transform 对应，不能简单拼接或平均两份 `norm_stats.json`。

联合 125 条数据的 norm 已按全部 285436 帧计算，包含 54 维 state 和 54 维 actions 的 mean、std、q01、q99：

```text
/home/descfly/lpy/openpi/assets/pi05_fr3_wuji_125ep/fr3_wuji/tomato_joint_125ep/norm_stats.json
```

计算时使用的配置：

```text
pi05_fr3_wuji_125ep
```

快速、向量化的计算脚本：

```bash
JAX_PLATFORMS=cpu uv run --no-sync examples/fr3_wuji/compute_norm_stats.py \
  --config-name pi05_fr3_wuji_125ep
```

脚本覆盖所有帧，并对每条 episode 的首、中、尾样本与正式 LeRobot 加载器进行一致性检查。

## 6. 训练配置和命令

联合训练配置名：

```text
pi05_fr3_wuji_125ep
```

主要参数：

- 基础权重：`pi05_base`
- LoRA 微调
- action dim：54
- action horizon：50
- batch size：8
- 总步数：20000
- 每 2000 步保存 checkpoint
- EMA：关闭
- WandB：关闭

训练命令：

```bash
cd /home/descfly/lpy/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_fr3_wuji_125ep \
  --exp-name tomato_lora_125ep_v2 \
  --log-interval 10
```

训练已完成，保存了 `10000` 和 `19999` 两个 checkpoint。最终权重：

```text
/home/descfly/lpy/openpi/checkpoints/pi05_fr3_wuji_125ep/tomato_lora_125ep_v2/19999
```

## 7. 离线回放评估

从旧、新数据各均匀选取 8 条 episode，每条在约 20%、50%、80% 位置取一个观测，共 48 个观测窗口。每个 checkpoint 使用相同采样噪声预测未来 50 步，并与示范动作比较。

这属于训练集离线回放，不是独立测试集，也不能代表真机任务成功率。

| 部位 | 10000 步 MAE | 19999 步 MAE | 误差降低 |
|---|---:|---:|---:|
| 左臂 | 0.0457 | 0.0297 | 35% |
| 左手 | 0.0512 | 0.0381 | 26% |
| 右臂 | 0.0228 | 0.0161 | 30% |
| 右手 | 0.0417 | 0.0303 | 28% |

整体 MAE 从 0.0433 降至 0.0312，降低约 28%。最终 `19999` 权重明显优于 `10000`。

评估产物：

```text
/home/descfly/lpy/openpi/replay_checks/joint125_eval_G9nAVm1s
```

对比图中的三条线：

- 蓝线：checkpoint 10000 与示范动作的平均绝对误差
- 橙线：checkpoint 19999 与示范动作的平均绝对误差
- 灰色虚线：未来保持当前关节位置时，与示范动作的误差基线

横轴是未来 50 步，约 0–1.63 秒；纵轴是关节位置平均绝对误差，单位为弧度。曲线表示误差，不是关节运动轨迹。

## 8. 动作跳变检查

最终权重在 48 个离线窗口中的最大变化：

| 部位 | 当前 state 到首个指令 | 50 步内部最大相邻变化 |
|---|---:|---:|
| 左臂 | 0.470 rad（26.9°） | 0.067 rad（3.8°） |
| 左手 | 0.370 rad（21.2°） | 0.110 rad（6.3°） |
| 右臂 | 0.093 rad（5.4°） | 0.025 rad（1.4°） |
| 右手 | 0.518 rad（29.7°） | 0.133 rad（7.6°） |

少数样本存在明显的首指令偏移和手部相邻指令跳变。上机时应先低速、限幅验证，重点关注：

- 当前状态到预测首帧的衔接
- 连续推理产生的 action chunk 之间是否平滑
- 机械臂和手指的每周期最大变化限制
- 急停和人工接管是否有效

## 9. 部署需要复制什么

最稳妥的方式是复制整个最终步骤目录：

```text
checkpoints/pi05_fr3_wuji_125ep/tomato_lora_125ep_v2/19999/
```

纯推理至少需要：

```text
19999/
├── params/
└── assets/
    └── fr3_wuji/
        └── tomato_joint_125ep/
            └── norm_stats.json
```

`train_state/` 只用于恢复训练，纯部署不需要。训练数据集也不需要复制到推理机器。

工控机或推理机的 OpenPI 代码必须包含 `pi05_fr3_wuji_125ep` 配置以及相同的 FR3/Wuji transform。

服务启动命令：

```bash
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_fr3_wuji_125ep \
  --policy.dir=/实际存放路径/19999
```

服务会从 checkpoint 内的 `assets/fr3_wuji/tomato_joint_125ep/norm_stats.json` 自动加载联合 norm。不能换成旧 65 条或新 60 条单独计算的统计。

## 10. 关键知识

1. LeRobot v2.1 只说明存储格式兼容，不能保证相机字段、state/action 维度、关节顺序和单位一致；训练前仍须逐项核对。
2. 原始数据、转换后的训练数据、norm 和 checkpoint 是四类不同资产，应分别保存和管理。
3. norm 必须基于本次训练实际使用的数据，并在与训练相同的 transform 之后计算。
4. 同任务的两个数据集可以联合训练；当前单数据集加载器要求先物理合并，或另外实现联合采样加载器。
5. 训练 loss 和训练集回放误差不能代替真机成功率。最终结论需要在安全限速条件下进行闭环真机测试。
6. 推理时必须携带与 checkpoint 对应的 norm；只复制参数会导致策略加载失败或动作尺度错误。
