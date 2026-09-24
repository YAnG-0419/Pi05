# 左手摘番茄强化训练

配置名：`pi05_fr3_wuji_left_enhance`。从 `pi05_base` 开始 LoRA 微调，使用约 300 条完整示范和从中裁出的 20 条左手阶段片段。现有加权采样器按帧数归一化每个数据集：完整数据占抽样期望的 80%，左手片段占 20%。这是整个训练过程的期望比例，单个 batch 不保证至少 20%。

## 数据准备

将两套数据分别保存为标准 LeRobot 数据集，或把服务器上的实际目录软链接到 LeRobot 缓存：

| 用途 | `repo_id` | 默认本地目录 |
| --- | --- | --- |
| 完整示范 | `fr3_wuji/tomato_full_300` | `~/.cache/huggingface/lerobot/fr3_wuji/tomato_full_300` |
| 左手阶段片段 | `fr3_wuji/tomato_left_20` | `~/.cache/huggingface/lerobot/fr3_wuji/tomato_left_20` |

左手片段作为独立 episode 保存，不修改或移除完整示范。每段从右手已拿起番茄、左手准备摘取的观测开始，覆盖左手摘取的完整动作。两套数据均须采用 30 Hz、`observation.state` 108 维、`action` 54 维，以及 `observation.images.cam0`、`cam1`、`cam2` 三路 RGB。state 和 action 保留现有 FR3/Wuji 原始顺序；在线 transform 负责切片和重排。所有 episode 的任务文本必须与部署时相同：

```text
Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.
```

数据就位后先运行：

```bash
uv run python -m examples.fr3_wuji.check_left_enhance_data
```

脚本会核对元数据、每条左手片段首/中/尾的图像解码和向量形状，并以固定种子抽样 20,000 次确认左手数据比例约为 20%。还需人工查看片段首尾画面，确认右手已拿起、左手动作完整、三个相机时间对齐；脚本只能检查可解码性和数据结构，不能判断动作语义。如果实际清洗结果的列名、维度或动作顺序不同，先修改数据转换或 policy transform，再计算统计量。

## 归一化与训练

新配置使用独立统计目录 `assets/pi05_fr3_wuji_left_enhance/fr3_wuji/tomato_full_300_left_20/`。两套数据实际内容确定后运行：

```bash
JAX_PLATFORMS=cpu uv run examples/fr3_wuji/compute_norm_stats.py \
  --config-name pi05_fr3_wuji_left_enhance
```

该脚本覆盖两套数据的全部帧，在 54 维在线变换后计算统计量；采样权重仅控制训练抽样，不用于统计量加权。脚本会拒绝覆盖已存在的统计文件。数据内容变化时，先移走旧统计目录，再重新计算；不要复用旧番茄模型的统计量。

先用新实验名做短程检查，再启动完整训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_fr3_wuji_left_enhance \
  --exp-name left_enhance_smoke --num-train-steps 20 --save-interval 10

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_fr3_wuji_left_enhance \
  --exp-name left_enhance_300plus20
```

正式配置沿用 batch size 8、20,000 步、每 2,000 步保存一次，默认不启用 WandB。检查短程训练的 loss 为有限数、checkpoint 已保存，并用该 checkpoint 推理确认输出动作形状为 `(50, 54)`。不要用 `--overwrite` 覆盖已有实验。

## 效果验证

在相同场景和任务指令下，分别记录现有模型与新模型的右手拿取、左手摘取成功率。左手成功率提高且右手表现未明显退步，才视为强化有效；训练 loss 不能替代真机评估。此次只训练混合组，不额外训练完整数据单独对照组。
