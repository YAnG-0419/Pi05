# 左手摘番茄强化训练（2026-09-26）

配置：`pi05_fr3_wuji_left_enhance`，从官方 `pi05_base` 开始 LoRA 训练，30 Hz、54 维动作、50 步动作窗口、batch size 8、20,000 步。当前分支 `left-train-enhance`，修改未经提交。

## 数据和采样

| 数据 | Episode | 帧数 | 抽样期望比例 | 实际目录 |
| --- | ---: | ---: | ---: | --- |
| 完整 A | 106 | 117995 | 53.33% | `/home/descfly/company-remote-2/real/harvest-tomato/datasets-pi05/0924-250ep/A` |
| 完整 B | 144 | 209919 | 26.67% | `/home/descfly/company-remote-2/real/harvest-tomato/datasets-pi05/0924-250ep/B` |
| 左手 A | 20 | 12858 | 13.33% | `/home/descfly/LTT/Left_only_exports/A` |
| 左手 B | 9 | 7737 | 6.67% | `/home/descfly/LTT/Left_only_exports/B` |

完整数据合计 80%，左手阶段合计 20%；各组内部 A:B=2:1。质量分级通过抽样次数体现，学习率共用。每个数据源按其帧数归一化，不保证逐 batch 比例。工程已在本地 LeRobot 缓存建立四个软链接，远端源数据未修改。PyAV 后端避开当前环境的 TorchCodec/FFmpeg 加载错误。

## 按顺序运行

2026-09-26 已完成数据预检和统计计算：348509 帧、837 个 LeRobot 对照样本通过。当前可直接从短程训练开始，以下预检/统计命令用于复查或数据更新后重算。

在仓库根目录执行。首次训练前重新计算四个来源的归一化统计量，统计量不按训练抽样权重加权：

```bash
cd /home/descfly/LTT/pi05
JAX_PLATFORMS=cpu uv run --no-sync python -m examples.fr3_wuji.check_left_enhance_data
JAX_PLATFORMS=cpu uv run --no-sync examples/fr3_wuji/compute_norm_stats.py --config-name pi05_fr3_wuji_left_enhance
```

统计目录为 `assets/pi05_fr3_wuji_left_enhance/fr3_wuji/tomato_full_250_left_29/`。统计脚本拒绝覆盖已有文件；再次训练相同数据不必重算。数据变化时先归档旧统计，再重新计算。

短程训练（与正式实验使用不同目录）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train.py pi05_fr3_wuji_left_enhance --exp-name left_smoke_0926 --num-train-steps 20 --save-interval 10 --log-interval 1
```

若短程 loss 有限且 checkpoint 保存成功，再正式训练：

```bash
uv run --no-sync wandb login
mkdir -p logs
set -o pipefail
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train.py pi05_fr3_wuji_left_enhance --exp-name left_enhance_0926 --wandb-enabled --log-interval 10 2>&1 | tee logs/left_enhance_0926.log
```

基础权重缺失时首次训练会自动下载。实验名已存在时换一个新名字，勿用 `--overwrite`。如需恢复正式实验，用相同命令加 `--resume`；20,000 是总训练步数。

## 查看过程和验证效果

启动正式训练后，打开终端打印的 WandB run 链接，在 Workspace 添加 `loss`、`grad_norm`、`param_norm` 图表，横轴选择 Step。loss 是训练集上的整体平均损失，尚未拆分 A/B 或左右手，不能单独证明左手改善。终端每 10 步也打印指标；另一个终端可运行 `tail -f logs/left_enhance_0926.log` 查看日志。

不连接 WandB 时去掉 `--wandb-enabled`，终端日志仍可用。`WANDB_MODE=offline` 可离线保存 WandB 日志，稍后联网通过 `uv run --no-sync wandb sync <offline-run目录>` 上传再查看曲线。

训练前人工检查裁剪片段的首尾和三路视频同步。最终用固定场景比较现有模型与新模型的右手拿取和左手摘取成功率。
