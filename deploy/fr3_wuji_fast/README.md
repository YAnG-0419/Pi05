# 19999 正常速度部署

独立入口，基于 Wuji 官方 RTG 的动作块管理方式适配本机 FR3 双臂 + Wuji 双手。
模型动作按 **30 Hz 原时间轴**播放，独立进程插值为 **100 Hz 设备下发**；不使用慢速播放器的
2.5 倍倍率、整段五次样条慢放、逐块停稳后再推理。首次接管仍受控接近模型首帧并确认到位。
详见 [上游来源及适配差异](UPSTREAM.md)。这不是未经修改的 Wuji 官方发行版。

模型服务、观测采集、FR3 网关/控制器、Wuji SDK 设备拥有者均导入或调用原有
19999 部署代码。共用接口新增会话级机械臂速度参数：快速入口显式传入 **1 rad/s**，
慢速及其他调用保留默认 **0.7 rad/s**；快速网关的最终容差为 **1.2 rad/s**，用于吸收100Hz
调度误差，快速轨迹目标仍限幅在1 rad/s。不修改全局速度常量、外部控制器或依赖。
两个执行入口共用原设备锁，不能同时占用硬件。

## 本机默认

| 项目 | 配置 |
| --- | --- |
| 左 / 右 FR3 | `172.16.0.2` / `172.16.1.2` |
| 左 / 右 Wuji | `192.168.1.110:7447` / `192.168.2.111:7447` |
| checkpoint | `/home/user/lpy/Pi05/checkpoints/19999_269/19999` |
| 模型服务 | `ws://127.0.0.1:8001` |
| 模型动作 | `50×54`，绝对关节位置，弧度 |
| 排列 | 左臂7、左手20、右臂7、右手20 |
| 图像 | cam0头部、cam1左腕、cam2右腕，沿用本机相机配置 |
| 下发 | 双臂 `ArmCommand` → 原网关；双手原 SDK `JointCommand` |
| 默认模式 | RTG、50次预测、执行结束后 disable |

`serve.sh` 复用原 `oneshot.policy_server`，加载 checkpoint 自带归一化并预热。
已有同一 checkpoint 的兼容服务在8001运行时可复用，客户端会核对模型、权重清单和归一化哈希。
若用另一份19999目录，服务和执行命令都要显式传同一个 `--checkpoint`。
30000的64维权重应使用其专用适配，本入口不支持混接。

## 检查与启动

均从 `/home/user/lpy/Pi05` 执行。

只检查文件、控制器构建和启动参数，**不会连接设备或模型**：

```bash
bash deploy/fr3_wuji_fast/run.sh --check
```

启动模型服务（不连接硬件）：

```bash
bash deploy/fr3_wuji_fast/serve.sh
```

服务就绪后，在另一个终端执行。**下面这条命令会启动真机控制并下发动作**，与慢速入口一样
使用现场值守模式；它不生成或声称通过物理停止验收。首次验证可用 `--rounds 1`，仍按原速播放：

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --execute --supervised-trial --start-cameras \
  --rounds 1 --finish-policy disable
```

连续 RTG 部署：

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --execute --supervised-trial --start-cameras \
  --broker-mode rtg --rounds 50 --finish-policy disable
```

已有相机发布时省略 `--start-cameras`。`--finish-policy hold` 会保持末姿态直到 Ctrl-C；
默认 `disable` 在最后实测到位后释放手部。异常停止仍使用原设备停止逻辑，不自动重连或续播。
`--read-only` 只读取设备反馈，需要已有双臂状态发布者。

`--broker-mode serial` 按50步播放完后才请求下一次推理，等待时持续发送末姿态；每段内部仍是30Hz。
RTG 默认 `--trigger-fraction 0.5 --guidance-steps 3`，动作块中途异步推理，并在新结果到达时
跳过推理期间已过去的节点、平滑接入剩余节点。`--rounds` 是接纳的预测次数，不保证每次都播完50步。
单块包含50个节点，首末节点相距49/30秒，最后节点保留一个周期，完整块占50/30秒。

## 继承的设备约束

本入口取消的是慢速规划的主体时间拉伸。原机械臂关节限位、1 rad/s变化率、0.08 rad运动跟踪、
接触力矩保护、20ms帧有效期、反馈新鲜度和独占控制保持。双手使用当前提交版本的slider模式：
1 rad/s指令限速、Kp=8、Kd=0.1、电流上限2A及已有接触处理；没有额外提高这些参数。
双手模型越界角度沿用慢速连续入口的投影规则，保存原始预测与投影数量。

任何机械臂节点段或RTG衔接段超过1 rad/s时，快速轨迹层逐关节按1 rad/s限幅并继续追赶目标，
日志记录被限幅的目标数量；原始模型预测仍保存在 `inference.npz`。它不改变30Hz时钟，也不会把
网关拒绝隐藏为执行成功。网关1.2 rad/s只提供下发时钟余量，不会主动生成更快轨迹。手部目标可因SDK限速而滞后，
所以30Hz计划不等于电机一定完整跟上30Hz轨迹。慢速试验成功不能证明这条原速路径已真机验证。

无硬件地检查历史预测（返回码1表示有预测被拒绝）：

```bash
bash deploy/fr3_wuji/run.sh -m deploy.fr3_wuji_fast.inspect \
  --record logs/weight_motion_eval/deployment-8edcf42e9b/round-0022/inference.npz
```

## 记录与离线验证

每次使用 `logs/weight_motion_eval/fast-<id>/` 新目录：

- `fast-config.json`、`runtime.json`：原速参数、设备、权重、继承限制。
- `chunk-NNNN/inference.npz`：原始50×54预测和输入实测状态。
- `chunk-NNNN/admission.json`：原始时间戳、请求ID、观测时效、任务指令。
- `control/events.jsonl`：RTG有效节点、跳过步数、实际下发目标、SDK限速输出和反馈。
- `fast-report.json`：完成/失败、接纳的动作块及原因；停止结果还需结合原桥接的 `device-exit.json`。
- 原始手部记录、网关记录和控制循环诊断继续由已有设备模块生成。

测试只使用虚拟设备；Unix socket与多进程测试也不连接硬件：

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider deploy/fr3_wuji_fast
```

覆盖30Hz节点和插值、RTG延迟补偿与实际衔接点、模型形状/单位契约、越界与超速拒绝、
动作块顺序/权重身份/原始期限、连续发送、推理缺失、调度超时、反馈过期、取消和断连停止。
另有真实本机socket + 独立控制进程测试，使用原 `DeviceSession` 与原 IPC bridge 验证双块执行及释放。

2026-09-14 抽查最近20条明确来自19999_269的历史预测，16条通过原始节点限位和机械臂原速检查，
4条超过0.7rad/s。该结果仅验证静态动作数组，不验证现场首帧、RTG衔接、碰撞、物理跟踪或停止。
本次开发没有启动真实机器人，也没有加载GPU模型。

速度参数更新后需重新运行快速执行命令，模型服务无需重启。
