# OpenPI FR3 与 Wuji 部署

更新日期：2026-10-17

本仓库用于 FR3 双臂与 Wuji 双手的 pi05 番茄采摘任务，包含模型微调、推理服务和机器人执行入口。基于 Physical Intelligence 的 OpenPI 项目，当前操作入口以本页和根目录的使用说明为准。

**完整操作说明：[FR3_Wuji_pi05部署使用说明.docx](FR3_Wuji_pi05部署使用说明.docx)**。文档包括快速指南、各模型启动命令、参数说明和state/action映射；其中Fast 1/3/30阈值表对应legacy行为，
2026-10-17后的19999 Fast默认值以本页、[快速部署 README](deploy/fr3_wuji_fast/README.md)和
[开发日志](docs/开发日志.md)为准。

> 2026-10-13：19999 快速入口已新增显式 RTC 模式；默认仍是 RTG。RTC 的最新接入边界、异机安装和
> 验证步骤以[快速部署 README](deploy/fr3_wuji_fast/README.md)与[开发日志](docs/开发日志.md)为准。
> 当前电脑只完成 Python 静态编译和 diff 空白检查，未运行 pytest、GPU/JAX JIT 或真机链路。

> 2026-10-17：19999 Fast的 `--arm-motion-limits` 默认 `disabled`；RTG、RTC、serial均取消正常模型回放的机械臂速度、加速度和jerk
> 软件上限；`--arm-motion-limits legacy` 恢复旧1 rad/s、3 rad/s²、30 rad/s³及网关1.2 rad/s。
> 30000v2 Fast包装默认仍为legacy。首次受控接近、关节位置、0.2 rad跟踪、接触力矩/电流相关保护、
> 20 ms时效、反馈健康/失联和急停均保留；新默认尚未完成目标工作站、ROS或真机验证。

## 快速开始

当前使用效果最好、建议优先使用 **70A_30B** 权重，模型为 **54 维、30 Hz、每次预测 50 步**。权重步骤目录为：

```text
checkpoints/pi05_fr3_wuji_weighted/tomato_lora_0918_a70_b30/19999
```

部署提供**慢、中、快三种执行速度**，另有独立 **Home 归位**。先启动模型服务，等待 `ready` / `Warmup OK`，再启动一个机器人控制端。以下示例针对已配置好的本机环境；迁移机器时需按使用说明核对路径、控制器与设备配置。

### 终端一 启动模型服务

```bash
cd /home/descfly/lpy/openpi
CKPT="$PWD/checkpoints/pi05_fr3_wuji_weighted/tomato_lora_0918_a70_b30/19999"
bash deploy/fr3_wuji/run.sh -m experiments.weight_motion_eval.oneshot.policy_server \
  --checkpoint "$CKPT" --port 8001
```

### 终端二 选择一种执行速度

先在第二个终端设置同一权重路径：

```bash
cd /home/descfly/lpy/openpi
CKPT="$PWD/checkpoints/pi05_fr3_wuji_weighted/tomato_lora_0918_a70_b30/19999"
```

以下命令包含 `--execute`，会驱动真机；只选择其中一条运行。

**慢速**：默认至少按模型时间轴的 2.5 倍时长播放。

```bash
bash deploy/fr3_wuji_slow/run.sh \
  --checkpoint "$CKPT" --uri ws://127.0.0.1:8001 \
  --execute --supervised-trial --continuous --rounds 50 --replan-steps 20 \
  --minimum-time-scale 2.5 --start-cameras --finish-policy disable
```

**中速**：显式使用 `--model 54`；默认约 25 秒后在下一段边界切慢速。全程中速可将 `--slow-after-seconds` 改为 `0`。

```bash
bash deploy/fr3_wuji_medium/run.sh --model 54 \
  --checkpoint "$CKPT" --uri ws://127.0.0.1:8001 \
  --execute --supervised-trial --continuous --rounds 50 --replan-steps 20 \
  --minimum-time-scale 0.625 --slow-after-seconds 25 \
  --start-cameras --finish-policy disable
```

**快速**：采用30 Hz模型时间轴，默认使用RTG异步衔接。下面的19999命令不传
`--arm-motion-limits`，因此正常模型回放默认没有本地机械臂速度、加速度或jerk软件上限；
这不表示硬件无限速或已通过真机验收。回退旧动态限值可增加 `--arm-motion-limits legacy`。

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --checkpoint "$CKPT" --uri ws://127.0.0.1:8001 \
  --execute --supervised-trial --broker-mode rtg --rounds 50 \
  --start-cameras --finish-policy disable
```

仅19999 / 50×54 / JAX 服务可显式选择 RTC。RTC 在模型端做 soft prefix guidance，执行端按请求到
实际接纳的延迟跳过过期节点，不再叠加 RTG cubic 拼接：

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --checkpoint "$CKPT" --uri ws://127.0.0.1:8001 \
  --execute --supervised-trial --broker-mode rtc \
  --rtc-initial-delay-steps 5 \
  --rtc-delay-buffer-size 10 \
  --rtc-prefix-attention-schedule exp \
  --rtc-max-guidance-weight 5 \
  --rounds 50 --start-cameras --finish-policy disable
```

迁移到另一台电脑时，确认workspace中的 `packages/openpi-client` 已更新且可导入
`openpi_client.rtc`。只有使用 `--arm-motion-limits legacy` 或默认legacy的30000v2 Fast包装时，
还需安装 `deploy/fr3_wuji_fast/requirements.txt` 中的Ruckig依赖；完整安装与无硬件验证命令见上述文档。

三种入口均以100 Hz下发设备指令。慢/中速以及Fast首次接管/legacy仍受各自轨迹约束；19999 Fast
默认回放不加动态软件上限，实际运动仍可能受底层控制器、硬件、推理衔接和到位等待影响。
慢/中速的 `--rounds 50 --replan-steps 20` 表示运行50轮，每轮预测50步、使用前20步；
快速入口不使用 `--continuous` 或 `--replan-steps`。

### Home 归位

在项目根目录执行：

```bash
# 只检查配置
bash deploy/fr3_wuji_home/run.sh --check

# 真机归位：先双臂，再双手，完成后释放退出
bash deploy/fr3_wuji_home/home.sh
```

Home 不需要模型服务或相机，不要与其他控制端同时运行。

## State 和 action 映射

硬件接口统一为 54 维，关节位置单位为 rad。索引从 0 开始，区间左闭右开。

| 部件 | 硬件 state / action 索引 | 维度 |
| --- | --- | --- |
| 左臂 | `[0:7]` | 7 |
| 左手 | `[7:27]` | 20 |
| 右臂 | `[27:34]` | 7 |
| 右手 | `[34:54]` | 20 |

输入 `observation/state` 为 `(54,)`，服务端输出 `actions` 为 `(50, 54)`，表示未来 50 步的绝对关节目标。70A_30B 服务已经完成反归一化与臂部 delta 还原，控制端不要再次叠加当前 state。

| 模型 | 模型内部 state 顺序 | 模型内部 action 顺序 |
| --- | --- | --- |
| 70A_30B / 原生 54 维 | 左臂→左手→右臂→右手 | 左臂→左手→右臂→右手 |
| 30000 / 30000v2，64 维 | 左臂→右臂→左手→右手→补零 10 维 | 左臂→右臂→左手→右手→末 10 维 |
| 25000 / 25000-single，64 维 | 左臂→左手→右臂→右手→补零 10 维 | 左臂→右臂→左手→右手→末 10 维 |

64 维模型由配套 server 补零、裁剪并重排，最终也返回硬件顺序的 `(50, 54)` 绝对关节目标。不同微调方式的模型可能需要让 **Codex 单独编写或调整 server 适配**，核对模型结构、输入输出维度、关节顺序、权重自带归一化统计、delta/绝对动作语义和模型频率。

切换权重时，先停止控制端和旧模型服务，再按模型身份同步切换服务入口、`--model`、`--checkpoint` 和端口。54/64 维及 30/20/15 Hz 的完整对应关系见使用说明和[模型适配说明](deploy/fr3_wuji_models/README.md)。

## 目录与文档

根目录的两份主要使用文档为本 README 和 Word 使用说明。历史说明与训练记录集中在 `docs/`；各代码模块的配套 README 保留在所属目录。

| 目录或文件 | 内容 |
| --- | --- |
| [FR3_Wuji_pi05部署使用说明.docx](FR3_Wuji_pi05部署使用说明.docx) | 当前部署操作说明，优先阅读 |
| [docs/README.md](docs/README.md) | 带整理日期的文档索引与历史资料 |
| `deploy/` | 模型服务、慢/中/快执行入口和 Home 归位 |
| `src/openpi/` | 模型、策略、数据变换与训练配置 |
| `scripts/` | 训练、数据准备与辅助工具 |
| `examples/` | 数据转换及其他机器人平台示例 |
| `checkpoints/` | 本地模型权重，使用完整步骤目录中的 `params/` 和 `assets/` |
| `experiments/weight_motion_eval/` | 推理、动作评估与部署相关实现 |
| `logs/` | 本机运行记录 |

训练与数据转换可从[番茄任务数据训练与部署记录](docs/Codex_番茄任务数据训练与部署记录.md)开始。通用安装、基础模型、其他机器人示例与 PyTorch 使用方式保留在[原 README 归档](docs/README_原版_2026-09-29.md)。历史资料中的机器路径和阶段性状态可能已变化，部署命令优先参考当前使用说明。

## 许可证

项目代码许可证见 [LICENSE](LICENSE)，Gemma 相关条款见 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。贡献流程见[贡献指南归档](docs/CONTRIBUTING_2026-09-29.md)。
