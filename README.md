# Franka 双臂 + Wuji 双手 DP 真机部署

工作目录：`/home/descfly/dp_infer`  
权重：`/home/descfly/dp_ckpt/dp/step=00010000.ckpt`

## 1. 配置与环境

配置文件：[configs/deploy.json](configs/deploy.json)。设备 IP 和模型端口沿用原 openpi 的 `fr3_wuji_fast` 部署：

| 服务/设备 | 地址 |
|---|---|
| DP 模型服务 | `http://127.0.0.1:8001` |
| 左 / 右 Franka | `172.16.0.2` / `172.16.1.2` |
| 左 / 右 Wuji | `192.168.1.110:7447` / `192.168.2.111:7447` |

部署机目录应为：

```text
/home/descfly/dp_infer/          本工程
/home/descfly/dp_ckpt/dp/        step=00010000.ckpt、SHA256SUMS.txt
/home/descfly/picking-tomatoes/  原控制源码
```

从开发机导出工程：

```bash
bash scripts/package.sh
```

将生成的 `dist/dp_infer-deploy.tar.gz` 传到部署机，在部署机解压：

```bash
tar -xzf dp_infer-deploy.tar.gz -C /home/descfly
cd /home/descfly/dp_infer
```

部署包不包含权重、ROS镜像、Wuji SDK、Python环境或旧编译产物。权重放到上表目录；ROS控制镜像和Wuji SDK沿用部署机已有环境。配置默认镜像为 `pico-armhand-teleop:latest`，SDK目录为 `/home/descfly/miniconda3/envs/pico-armhand-teleop/lib/python3.10/site-packages`。若实际名称/位置不同，在 `configs/deploy.json` 修改 `docker_image`、`sdk_site`、`reference`；SDK须与Humble容器的Python 3.10兼容。

模型环境任选一种准备方式：

```bash
# 新建项目内模型环境，需要部署机已安装Python 3.10及venv
bash scripts/setup_env.sh
```

已有DP模型环境时，在每个运行终端设置 `DP_PYTHON` 为该环境Python的绝对路径，跳过安装。`run.sh` 优先使用 `DP_PYTHON`，否则使用项目 `.venv/bin/python`。原pi0.5环境不一定包含本模型的PyTorch依赖。

ROS默认 `ROS_DOMAIN_ID=0`、`ROS_LOCALHOST_ONLY=0`、`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`，须与现场相机节点一致。配置里的CPU分配也须按部署机实际核数调整，左右控制器与其他进程使用互不重叠的CPU集合。

**首次迁移必须在部署机重新构建控制器**，不能直接复用其他目录或机器的 `.deployment`。随后运行环境检查：

```bash
./run.sh build --fresh
./run.sh doctor --container --checkpoint
```

`--fresh` 会备份旧overlay再重建；`doctor` 应输出 `"all_passed": true` 后再启动模型。检查失败时先修正其提示的镜像、SDK、GPU、CPU或路径配置。以上命令均不连接机器人。

## 2. 准备设备和相机

1. 在 Franka Desk 完成解锁、故障处理和 FCI 激活，双手上电。
2. 将双臂双手放到接近训练场景的起始姿态并保持静止；程序不会自动回零或恢复初始姿态。
3. 退出原 pi0.5 模型服务、遥操、旧策略执行器、双臂控制器和手部 SDK 程序，保留现场相机节点。设备只能由一套控制程序接管。
4. 确认以下相机话题持续发布，视角对应正确：

| 观测 | ROS 话题 |
|---|---|
| 头部 RGB | `/cam0/color/image_raw` |
| 左腕 RGB | `/cam1/color/image_raw` |
| 右腕 RGB | `/cam2/color/image_raw` |
| **头部深度（必需）** | **`/cam0/depth/image_raw`** |

深度格式为 `16UC1`（默认毫米）或 `32FC1`（米）。若原 pi0.5 相机配置关闭了深度，先在原相机配置中开启 cam0 深度。本工程复用已有相机节点，不负责启动相机。

启动前可查看剩余容器和端口占用：

```bash
docker ps --format 'table {{.Names}}\t{{.Command}}'
ss -ltnp 'sport = :8001 or sport = :8876'
```

8001 应已由旧模型释放。8876 是客户端自动管理的内部观测端口，也应空闲。DP 使用 HTTP 协议，必须搭配本目录客户端，不能使用原 openpi WebSocket 客户端。

## 3. 启动模型服务——终端 A

```bash
cd /home/descfly/dp_infer
./run.sh serve --host 127.0.0.1 --port 8001
```

等待权重加载和预热完成，出现以下提示后保持该终端运行：

```text
Policy ready: http://127.0.0.1:8001
```

## 4. 启动真机执行——终端 B

**以下命令会启动双臂控制器、使能双手并下发动作。现场需有人值守并可操作物理急停。**

首次先执行一块动作：

```bash
cd /home/descfly/dp_infer
./run.sh client --execute --supervised-trial --rounds 1
```

客户端会自动启动双臂控制器、网关、Wuji 设备桥和观测采集，读取实测状态与 RGBD 后推理并执行，无需手动启动这些组件。

确认运动方向、跟踪和停止正常后，再启动多块闭环：

```bash
./run.sh client --execute --supervised-trial --rounds 10
```

- `--rounds` 是预测动作块数，每块包含8步动作。每块到位后读取新观测并重规划。
- 当前按速度/加速度约束逐块播放，设备下发频率100 Hz，动作时间会延长。
- 默认总执行时限120秒；需要更长运行时间时修改 `configs/deploy.json` 的 `max_run_s`。
- 正常结束会释放双手并关闭本次设备进程；手中物体需有承接。程序不会自动返回起始姿态。
- 每条执行命令是独立会话，等待前一条完全退出后再启动下一条；模型服务可以持续运行。

## 5. 停止与查看结果

中途停止时，在**终端 B 按 `Ctrl-C`**，等待设备停止和进程清理；随后可在终端 A 按 `Ctrl-C` 停止模型服务。出现即时物理风险时使用现场急停或 Franka Desk。

每次运行记录保存在终端打印的 `Saved: ...` 目录（默认 `logs/dp-时间-标识/`）：

| 文件 | 用途 |
|---|---|
| `report.json` | 完成状态、已执行块数和错误；正常完成为 `state=completed` |
| `device-exit.json` | 设备退出状态、故障及停止错误 |
| `chunk-*/prediction.npz` | 每块模型原始预测 |
| `control.npz` | 实际下发目标和实测反馈 |
| `arms.log` / `devices.log` / `observations.log` | 控制器、双手与观测故障定位 |

启动阶段失败时可能没有 `report.json`，直接查看同目录的子进程日志。再次启动前确认设备已停稳、旧设备进程已退出、故障已处理，并重新确认当前姿态和夹持状态。

| 常见问题 | 处理 |
|---|---|
| 8001/8876被占用 | 退出已确认的旧模型或旧DP客户端，保留相机服务 |
| 缺少完整观测 | 检查cam0深度、三路RGB、ROS domain及设备反馈；查看 `observations.log` |
| 已有设备owner | 退出原遥操、双臂控制器、直接FCI读取器和手部SDK程序 |
| 首步距离过大或关节越限 | 核对起始姿态、左右设备与相机映射，不跳过限位检查 |
| 控制超时或无法到位 | 停止后检查设备日志、CPU调度、反馈和机械阻滞 |
