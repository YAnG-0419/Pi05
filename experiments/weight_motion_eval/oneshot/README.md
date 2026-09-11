# 单次 50 步播放器：开发状态

2026-09-10。目标是现场新观测 → 权重 19999 推理一次 → 首步接近 → 到位确认 → 播放完整 50 个节点 → 结束确认。没有周期重推理，也不会恢复已暂停的旧轨迹。

**统一真实设备接口和部署入口已经接通；默认检查模式不启动设备。** 常规 `--execute` 需要匹配当前设备与控制器版本的实测验收记录；现场值守的首次单次试运行可显式使用 `--execute --supervised-trial`，跳过历史验收文件要求。这个模式不会生成或声称已经通过物理停止验收。

## 已完成

- 手部规划速度上限 30°/s；首步接近进一步降到 15°/s。预测过快时规划器拉长整段时间，SDK 提交边界按实际发送间隔限制位置增量，目标增量过快会截限后发送，同时保留实测速度检查；速度字段保持零，不能用 MIT 的目标速度字段替代限速。
- 机械臂沿用原网关的 0.7 rad/s、0.05 rad 接管阈值和接触力矩保护。规划仍使用更低的首步 0.15 rad/s、播放 0.35 rad/s。
- `core.py`：单次计划状态机、计划指纹、起始实测姿态、反馈新鲜度、100 Hz 调度、跟踪误差、暂停与故障停止。模拟中到位需要位置误差 ≤0.01 rad、速度 ≤0.02 rad/s 持续 0.5 秒。
- 每个输出帧独立携带最长 20 ms 截止时间。`transport.py` 与 `ros_boundary.py` 保留原始时间戳；双臂有一侧拒绝时不转发另一侧。
- `prepare_overlay.py` 将原机械臂控制器复制到本仓库 `.deployment/oneshot-overlay/`，加入实时控制循环中的到期检查和故障保持。已经编译。原参考仓库没有修改。
- `hand.py`：显式 SDK 所有权、20 个关节 ID 映射、实测状态与诊断读取、发送检查和急停请求。常规模式使能前匹配实测验收记录；现场试运行模式绑定本次连接读回的设备序列号、固件、增益和电流限制，使能前再次核对，参数变化仍会拒绝。
- `devices.py`、`ros_devices.py`、`bridge.py`：双臂与双手统一会话。只有网关向 FR3 命令总线发布；先等待本帧双臂网关确认，再提交双手，部分提交失败也会中止。设备进程独立检查命令截止时间、客户端断开和反馈健康状态。
- `ipc.py`、`live.py`、`deploy.py`：主机的模型/规划环境与 ROS/SDK 容器通过本机 Unix 消息套接字连接。现场相机和实测状态只推理一次；拒绝过期返回，绝不加载历史记录当现场指令。SDK 进程发布手部实测 ROS 状态供采集使用，采集端不再建立第二份 SDK 连接。
- 手部使能结束后重新检查姿态和速度，再开始轨迹时钟。正常完成默认保持末姿态，设备进程维持手部保持并接收主机监测心跳；Ctrl-C 请求停止并在实测停稳后释放。也可明确选择 `--finish-policy disable`。

30°/s 是软件指令约束：`q_send = q_last_sent + clip(q_target - q_last_sent, -v_max * dt, v_max * dt)`，其中 `v_max = π/6 rad/s`。100 Hz、间隔恰好 10 ms 时，每关节最多变化 0.3°；使用实际发送间隔，断流超过 30 ms 不补发追赶。限幅后的目标仍受原位置、跟踪和命令有效期检查约束。完整规划统一放慢以保持双臂和双手同步，SDK 限幅仅作为发送边界的补充；端点仍需实测到位才算完成。

限制目标位置变化不能保证真机速度始终不超过 30°/s；电机跟踪瞬态、振荡或反馈异常仍可能触发实测超速停止。该保护没有改成忽略超速。没有修改手部 MIT 增益、电流限制、固件或故障复位行为。

2026-09-10 30°/s 更新验证：94 项回归通过，涵盖 50°/s 目标自动限幅、正反向 20 关节、5/15 ms 交替发送间隔、最终目标到达、发送失败不推进限速器、过期/断流/实测超速停止。使用最近一次真机保存的预测仅作离线模拟，完成 1596 帧、约 15.96 秒；首步接近约 3.811 秒、主体播放约 11.104 秒。结果：`logs/weight_motion_eval/hand-speed30-20260910/`。本轮没有连接或使能真机，不表示已经解决之前的实测超速或红灯故障。

## 可运行入口

在仓库根目录执行，输出必须使用一个不存在的新目录：

```bash
bash experiments/weight_motion_eval/run.sh oneshot-simulate \
  --record logs/weight_motion_eval/new19999-restored-20260910/live/inference-0000.npz \
  --output logs/weight_motion_eval/my-oneshot
```

输出 `plan.npz`、`report.json`、`preview.html`、`oneshot.json`、`oneshot.npz`。可追加 `--inject feedback_loss`、`consumer_delay`、`scheduler_delay`、`pause` 或 `partial_submit` 检查停止路径。所有这些命令只运行模拟设备。

历史文件明确标为历史输入；模拟使用虚拟时钟，不会把旧观测重新标记为现场有效。真实实现需在推理返回时验证原观测期限 200 ms、发送时 70 ms，再把预测一次性接纳为有限时长计划；后续插值帧的 20 ms 截止时间是另一层检查。

## 本轮验证

结果目录：`logs/weight_motion_eval/oneshot-development-20260910/`。

- 更新参数后，复原姿态采集的 145 条历史预测全部通过连续曲线规划检查。
- 第一条预测：首步接近 2.765 秒，模型播放 12.264 秒，倍率 7.509；模拟含到位等待共 16.07 秒，输出 1607 个插值帧。
- 原生 C++ 截止时间检查已编译测试；隔离控制器包编译通过。
- Docker `--network none`、ROS domain 197 内，使用假反馈测试实际 ROS 消息传输：正常双臂转发并保持时间戳、过期后故障锁存、缺失力矩反馈拒绝、过期帧拒绝、单臂不合格时整组拒绝。未启动硬件驱动。
- Python 回归覆盖原规划器、单次播放器、SDK 最终发送边界和异常路径。运行方式：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. .venv/bin/python -m pytest -q \
  experiments/weight_motion_eval/test_motion_eval.py \
  experiments/weight_motion_eval/oneshot/test_core.py \
  experiments/weight_motion_eval/oneshot/test_boundaries.py
```

这些验证不等于物理停止、碰撞路径或实际任务成功验收。

接口集成后的新增结果位于 `logs/weight_motion_eval/deployment-interfaces-20260910/`：

- 88 项 Python 回归测试通过，包括原观测/推理记录代码的兼容性检查。
- `full-stack-02/` 与 `full-stack-hold-03/`：真实主机规划、IPC、ROS 网关与分发器连接模拟设备，分别验证结束释放、结束保持。各完成 1607 帧，平均约 99.999 Hz，最长输出间隔分别约 11.36 ms、11.09 ms。没有 FCI 驱动或真实 SDK 设备参与这些模拟。
- 模拟中先发现同步反馈读取的重复开销导致超期，优化了读取开销后通过；20 ms 命令寿命和超期中止检查保留。
- 隔离控制器的 ROS 启动入口已通过 `--show-args` 检查，没有启动驱动。原 `robot_control.launch.py` 会引入一个直接写控制器的复位节点，因此新入口使用内部的 `franka_fr3_arm_controllers.launch.py`，保留碰撞阈值配置与控制器启动顺序，单次运行期间不提供复位服务。
- `read-only-devices-02/`：新设备进程实际连接左右手 SDK，两个手部状态均收到且检查时无反馈错误；当前双臂 ROS 实测消息缺失，因此四设备预检按预期失败退出。没有使能/禁用手部或发送关节目标，退出后无本次 SDK 所有权残留。

## 真实设备部署入口

默认地址已按工作区设备设置，可用同名参数覆盖：

| 设备 | 地址 |
| --- | --- |
| 左臂 | `172.16.0.2` |
| 右臂 | `172.16.1.2` |
| 左手 | `192.168.1.110:7447` |
| 右手 | `192.168.2.111:7447` |

仅生成配置与待执行命令，不启动容器或连接设备：

```bash
bash experiments/weight_motion_eval/oneshot/run.sh --check \
  --left-arm-ip 172.16.0.2 --right-arm-ip 172.16.1.2 \
  --wuji-sides both \
  --wuji-left-address 192.168.1.110:7447 \
  --wuji-right-address 192.168.2.111:7447 \
  --output logs/weight_motion_eval/my-deployment-check
```

输出 `workcell.yaml`、`runtime.json` 和 `commands-preview.json`，同时校验控制器源文件/产物、参考限位、权重与归一化文件。所有输出目录必须尚不存在。

连接双手 SDK 并读取四组反馈，不使能、不发送控制目标：

```bash
bash experiments/weight_motion_eval/oneshot/run.sh --read-only \
  --output logs/weight_motion_eval/my-readonly
```

只读模式不启动 FCI 控制器，需要已有的双臂 ROS 实测状态；没有双臂反馈就会失败并清理自己启动的设备进程。双手只允许本进程持有 SDK，已运行的 GUI/手部控制进程需要先正常退出。读到的设备身份和参数会先写入 `devices-inventory.json`。

权重 19999 的专用模型服务，使用该检查点自己的归一化文件，并在接受客户端前预热：

```bash
bash deploy/fr3_wuji/run.sh -m experiments.weight_motion_eval.oneshot.policy_server \
  --checkpoint checkpoints/19999 --port 8001
```

它继续使用现有 `pi05_fr3_wuji` 训练配置的动作变换：臂部 delta 转回绝对关节位置，手部绝对关节位置。客户端校验检查点清单和归一化文件的 SHA-256，防止连到旧权重服务。

现场值守的首次单次试运行（用户已明确允许暂免历史验收文件）：

```bash
bash experiments/weight_motion_eval/oneshot/run.sh --execute --supervised-trial \
  --checkpoint checkpoints/19999 --uri ws://127.0.0.1:8001 \
  --start-cameras --finish-policy hold \
  --output logs/weight_motion_eval/my-supervised-live-50
```

仅暂免历史验收文件；设备独占、反馈新鲜度、关节范围、机械臂原网关、手部 30°/s、命令截止时间及停止路径保持启用。单次新推理 50 步，首步接近和后续插值沿用现有规划。试运行不设置额外的手部首步角度差门槛，完整曲线仍需通过关节范围、速度、加速度及总时长检查。`execution-admission.json` 明确记录豁免，`trial-device-readback.json` 保存本次真实读回参数。该选项必须与 `--execute` 同用，不能与 `--qualification` 同用。

完成现场验收并保存记录后，常规单次真实部署命令为：

```bash
bash experiments/weight_motion_eval/oneshot/run.sh --execute \
  --checkpoint checkpoints/19999 --uri ws://127.0.0.1:8001 \
  --qualification logs/weight_motion_eval/commissioning/qualification.json \
  --start-cameras --finish-policy hold \
  --output logs/weight_motion_eval/my-live-50
```

`--execute` 会启动本次隔离双臂控制器、网关、分发器和双手 SDK 进程。它拒绝与已有的 FCI/SDK 所有者同时运行，不会停止用户原有的进程。`--start-cameras` 仅在相机未运行时使用。摄像头预热期间没有预测动作输出；完整 50 步执行结束后不会再推理下一段。

`events.jsonl` 的 `command` 为规划目标，`hand_output` 保存 SDK 实际提交的左右手位置、被截限的关节索引（各手 0–19）、发送时间与限速时间间隔。

日志包含 `live-inference.npz`、`plan/`、`events.jsonl`、`live-report.json` 和各进程日志。`stop-request.json` 只记录停止请求；实测停止和资源释放结果看 `device-exit.json`，尤其是 `physical_stop_confirmed`、`hands_still_owned`、`stop_errors`。这些字段不表示任务成功或避碰通过。

## 仍需现场完成的验收

2026-09-10 首次现场试运行结果：`logs/weight_motion_eval/supervised19999-20260910-01/live/`。双臂控制器成功激活，双手使能，现场推理往返 125.6 ms。首步接近成功提交 42 个插值帧（序号 0–41），跨度 0.410 秒、平均 99.92 Hz，随后实测关节速度检查触发停止，未进入 50 节点主体播放。`device-exit.json` 确认停稳、双手所有权已释放、无停止错误；本次控制容器均已退出。计划接近 2.493 秒、主体播放 11.194 秒，但均未执行完成。原始异常未记录触发关节及速度，不能从最后一帧成功记录确定是哪一关节；已补充后续异常的关节索引、实测值和阈值记录，没有放宽速度检查或自动重试。具体指标见 `trial-summary.json`。

1. 确认两只手断流、执行进程退出后的实际停止行为，以及从禁用到使能不会恢复旧目标。之前只读查询发现左手固件 2.2.1、右手 2.2.3，都有 `emergency_stop` 接口，未发现可直接确认的断流看门狗资源。软件急停 RPC 成功不等于网络中断时仍能停车。
2. 验证隔离控制器实际跟踪和停止效果，以及真实 ROS/SDK 负载下的时间预算。
3. 完成手部首步接近范围与运行参数验收。之前只读参数均为 kp=8、kd≈0.1、电流限制 1 A；部署入口不改写这些参数。验收记录必须明确 `hand_raw_initial_delta_rad`，真实规划会据此拒绝超范围接近。

`qualification.py` 读取的记录结构见 [qualification.example.json](qualification.example.json)。示例是未验收模板，不能执行；它绑定控制器二进制哈希、双手身份及运行参数、每项实测证据文件的哈希。不能为了启动而直接把示例中的 `passed` 改成 true。记录检查只能防止错用/变更证据，不能替代测量本身。
