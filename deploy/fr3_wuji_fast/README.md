# 19999 正常速度部署（RTG / RTC / serial）

独立入口，基于 Wuji 官方 RTG 的动作块管理方式适配本机 FR3 双臂 + Wuji 双手。
模型动作按 **30 Hz 原时间轴**播放，独立进程插值为 **100 Hz 设备下发**；不使用慢速播放器的
2.5 倍倍率、整段五次样条慢放、逐块停稳后再推理。首次接管仍受控接近模型首帧并确认到位。
详见 [上游来源及适配差异](UPSTREAM.md)。这不是未经修改的 Wuji 官方发行版。

本入口新增显式 `--broker-mode rtc`：模型端用 soft prefix guidance 让新动作块对齐尚未执行的
旧动作前缀，客户端再按本次推理的**实际单调时钟延迟**跳过已经过期的节点。RTC 是推理时引导，
不训练或替换 checkpoint，也不等同于执行端的运动限值。RTC 接入时不再叠加 RTG 的 cubic 前缀拼接。

模型服务、观测采集、FR3 网关/控制器、Wuji SDK 设备拥有者均导入或调用原有
19999 部署代码。`--arm-motion-limits` 可选 `disabled` / `legacy`；19999 Fast默认值为
`disabled`。在不传该参数时，RTG、RTC、serial三种broker
都按模型的30 Hz原时间轴回放，默认不再对正常模型回放增加机械臂关节速度、加速度或jerk软件上限，
也不使用原来的1 rad/s轨迹投影、Ruckig 1/3/30约束或网关1.2 rad/s变化率阈值。
需要回退旧行为时显式传 `--arm-motion-limits legacy`。该选项恢复1 rad/s速度、3 rad/s²加速度、
30 rad/s³ jerk和网关1.2 rad/s阈值。慢速、中速以及30000v2 Fast专用包装的默认值不随此改变；
30000v2 Fast包装默认仍选择 `legacy`。两个执行入口共用原设备锁，不能同时占用硬件。

这里取消的是**正常模型回放阶段的本地机械臂动态软件限值**，不是取消全部保护，更不表示机械臂
在物理上“无限速”。首次接管仍使用受控接近；关节位置范围、0.2 rad跟踪误差、接触力矩与手部
电流相关保护、20 ms帧有效期、反馈健康/失联处理、独占控制、故障停止与急停链路均保留。
底层FR3控制器和硬件自身还可能有独立限制。本默认路径尚未完成目标工作站、ROS链路或真机验证。

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
| 默认模式 | RTG、50次预测、执行结束后 disable（RTC 必须显式选择） |
| 19999 Fast机械臂动态限值 | 默认 `disabled`，不对正常回放增加速度/加速度/jerk软件上限；`--arm-motion-limits legacy` 恢复旧1/3/30及网关1.2 |
| RTC 默认值 | 初始延迟5步、延迟历史容量10、`exp` schedule、最大引导权重5 |
| RTC 支持范围 | 仅19999、`50×54`、JAX 推理路径 |

`serve.sh` 复用原 `oneshot.policy_server`，加载 checkpoint 自带归一化并预热。
已有同一 checkpoint 的兼容服务在8001运行时可复用，客户端会核对模型、权重清单和归一化哈希。
若用另一份19999目录，服务和执行命令都要显式传同一个 `--checkpoint`。
30000的64维权重应使用其专用适配，本入口不支持混接。RTC 模式还会在启动设备前核对服务端
协议版本、`soft_prefix_guidance` 能力、schedule 清单和 guided JIT 预热状态；旧服务、PyTorch
路径及64维模型都不会被当作 RTC 服务使用。

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
使用现场值守模式；它不生成或声称通过物理停止验收。命令不带 `--arm-motion-limits`，因此正常
模型回放默认没有本地机械臂速度、加速度或jerk软件上限。当前尚无真机验收结果，不能把该默认值
直接当作首次真机试验的安全配置。首次验证可用 `--rounds 1`，仍按原速播放：

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --execute --supervised-trial --start-cameras \
  --rounds 1 --finish-policy disable
```

需要复现修改前的动态限值时，给任一 `rtg` / `rtc` / `serial` 命令增加：

```bash
--arm-motion-limits legacy
```

这会同时恢复机械臂1 rad/s、Ruckig加速度3 rad/s²、jerk 30 rad/s³和网关1.2 rad/s阈值；
它不改变模型、RTC引导或30 Hz时间轴。

连续 RTG 部署：

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --execute --supervised-trial --start-cameras \
  --broker-mode rtg --rounds 50 --finish-policy disable
```

连续 RTC 部署（仅19999 / 54D / JAX）：

```bash
bash deploy/fr3_wuji_fast/run.sh \
  --execute --supervised-trial --start-cameras \
  --broker-mode rtc \
  --rtc-initial-delay-steps 5 \
  --rtc-delay-buffer-size 10 \
  --rtc-prefix-attention-schedule exp \
  --rtc-max-guidance-weight 5 \
  --rounds 50 --finish-policy disable
```

已有相机发布时省略 `--start-cameras`。`--finish-policy hold` 会保持末姿态直到 Ctrl-C；
默认 `disable` 在最后实测到位后释放手部。异常停止仍使用原设备停止逻辑，不自动重连或续播。
`--read-only` 只读取设备反馈，需要已有双臂状态发布者。

`--broker-mode serial` 按50步播放完后才请求下一次推理，等待时持续发送末姿态；每段内部仍是30Hz。
RTG 默认 `--trigger-fraction 0.5 --guidance-steps 3`，动作块中途异步推理，并在新结果到达时
跳过推理期间已过去的节点，再用 cubic 前缀平滑接入剩余节点。RTC 也在块中途请求，但会把
请求发出时仍有效的旧动作作为 soft prefix 交给模型；控制进程接纳结果时用请求发出至接纳的实际
端到端延迟向上取整计算跳步，直接安装模型已经引导过的剩余动作，**不会再运行 cubic 拼接**。
RTG、RTC和serial共用同一机械臂运动限值模式：19999 Fast默认直接使用上述时间轴结果，
不再经过1 rad/s投影或Ruckig；只有 `--arm-motion-limits legacy` 才恢复旧动态限制。
三种模式都继续经过关节位置、跟踪、接触、时效、反馈健康与停止检查。
`--rounds` 是接纳的预测次数，不保证每次都播完50步。
单块包含50个节点，首末节点相距49/30秒，最后节点保留一个周期，完整块占50/30秒。

RTC 延迟历史以 `--rtc-initial-delay-steps 5` 为种子，首块普通推理的实测延迟也会先写入历史；
第一次 RTC 请求即从容量为 `--rtc-delay-buffer-size 10` 的当前历史中取保守最大值作为模型条件。
之后每个已接纳 RTC 块只记录一次请求发出至控制进程安装的端到端 offset；同一 RTC 轮次不会再让
WebSocket RTT 与安装 offset 分别占用两个槽位。历史在首轮仍包含 vanilla RTT，随后逐步由已接纳
RTC 块的端到端 offset 滚动替换，容量10最多保留10条样本。预测值与接纳时按本次实测延迟计算的
跳步是两个值。`--rtc-prefix-attention-schedule exp` 和
`--rtc-max-guidance-weight 5` 控制模型端前缀权重。线上协议字段分别为
`inference_delay_steps`、`prefix_attention_horizon`、`prefix_attention_schedule`、
`max_guidance_weight`；服务响应把回执放在顶层 `rtc` 字段。请求和回执还绑定源动作块编号、哈希和
token。过期前缀、回执不匹配或实测延迟超过已引导 horizon 都会中止整次执行并走既有设备停止路径，
不是只拒绝这次切换后继续播放旧块。

## 继承的设备约束

本入口除取消慢速规划的主体时间拉伸外，19999 Fast默认还取消正常模型回放的机械臂1 rad/s
变化率投影、Ruckig 1/3/30和网关1.2 rad/s变化率阈值。首次接管仍从实测姿态通过原有受控接近
到模型首帧；机械臂关节位置范围、0.2 rad运动跟踪、接触力矩保护、20 ms帧有效期、反馈新鲜度、
失联停止、独占控制、故障停止和急停保持。双手使用当前提交版本的slider模式：
1 rad/s指令限速、Kp=8、Kd=0.1、电流上限2A及已有接触处理；没有额外提高这些参数。
快速机械臂跟踪误差参数位于 `deploy/fr3_wuji_fast/limits.py`：
`ARM_TRACKING_TOLERANCE_RAD = 0.2`（单位 rad）。快速控制循环及设备下发检查共用此值，
运行时写入 `runtime.json` 的 `arm_tracking_rad`。修改后重启快速执行端即可，模型服务无需重启。
此参数也用于复用快速执行器的64维适配入口；慢速部署仍使用原来的0.08 rad默认值。
双手模型越界角度沿用慢速连续入口的投影规则，保存原始预测与投影数量。

默认模式下，相邻机械臂模型节点及RTG/RTC衔接产生的变化不会再被本地动态限值改写；30 Hz节点
仍线性插值为100 Hz命令。`--arm-motion-limits legacy` 才会重新投影目标并使用Ruckig和1.2 rad/s
网关阈值。原始模型预测仍保存在 `inference.npz`。手部目标可因SDK限速而滞后，所以30 Hz计划
不等于电机一定完整跟上30 Hz轨迹。无软件动态上限也不代表底层控制器没有限制；慢速或legacy试验
成功都不能证明新的默认路径已经过真机验证。

无硬件地检查历史预测（返回码1表示有预测被拒绝）：

```bash
bash deploy/fr3_wuji/run.sh -m deploy.fr3_wuji_fast.inspect \
  --record logs/weight_motion_eval/deployment-8edcf42e9b/round-0022/inference.npz
```

## 记录与离线验证

每次使用 `logs/weight_motion_eval/fast-<id>/` 新目录：

- `fast-config.json`：RTC 配置位于 `options.rtc_initial_delay_steps`、
  `options.rtc_delay_buffer_size`、`options.rtc_prefix_attention_schedule`、
  `options.rtc_max_guidance_weight`；运行配置同时以 `arm_motion_limits`、`arm_speed_enforced`、
  `arm_speed_rad_s` 和 `arm_smoothing` 记录所选机械臂运动限值模式。`runtime.json` 保存
  设备、权重和仍生效的边界，不重复 RTC 上下文。审计时必须区分默认 `disabled` 和 `legacy`，
  不能从旧日志中的1/3/30字段推断新运行仍在限速。
- `chunk-NNNN/inference.npz`：原始50×54预测、输入实测状态；RTC 请求还以
  `rtc_previous_actions` 保存送入模型的旧动作前缀。
- `chunk-NNNN/admission.json`：原始时间戳、请求ID、观测时效、任务指令；RTC 字段位于 `rtc` 对象，
  完整键为 `protocol_version`、`token`、`source_number`、`source_sha256`、
  `inference_delay_steps`、`actual_websocket_delay_steps`、`prefix_attention_horizon`、
  `prefix_attention_schedule`、`max_guidance_weight`、`receipt`。
- `control/events.jsonl`：RTG/RTC有效节点、跳过步数、实际下发目标、SDK限速输出和反馈；RTC 换块事件的
  `rtc` 对象完整键为 `token`、`source_number`、`source_sha256`、`predicted_delay_steps`、
  `actual_delay_steps`、`prefix_attention_horizon`、`prefix_attention_schedule`、
  `max_guidance_weight`。
- `fast-report.json`：完成/失败、接纳的动作块及原因；`chunks[].rtc` 复制上述换块字段。停止结果还需
  结合原桥接的 `device-exit.json`。
- 原始手部记录、网关记录和控制循环诊断继续由已有设备模块生成。

现有测试设计只使用虚拟设备；Unix socket与多进程测试也不连接硬件。以下命令留给具备完整环境的
工作站执行：

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -p no:cacheprovider deploy/fr3_wuji_fast
```

覆盖30Hz节点和插值、RTG/RTC延迟补偿与实际衔接点、RTC 回执绑定及不重复 cubic 拼接、
模型形状/单位契约、默认动态目标不被改写、legacy 1/3/30回归、关节越界与其余安全边界拒绝、
动作块顺序/权重身份/原始期限、连续发送、推理缺失、调度超时、反馈过期、取消和断连停止。
另有真实本机socket + 独立控制进程测试，使用原 `DeviceSession` 与原 IPC bridge 验证双块执行及释放。

截至 **2026-10-17**，RTC与机械臂运动限值模式的代码/离线验证结果以
[`docs/开发日志.md`](../../docs/开发日志.md)最新条目为准。当前仍未完成目标工作站、GPU/JAX、
ROS整链路或真机验证，本文不声称新的默认回放路径已通过实机验收。建议按以下顺序验证：

1. 在仓库根目录安装锁定依赖；根工作区会以 editable workspace 包安装
   `packages/openpi-client`，因此另一台电脑必须同步包含本次新增的 `openpi_client.rtc`。
   只有需要 `--arm-motion-limits legacy`（30000v2 Fast包装默认使用）时，才需要安装快速入口中
   固定的Ruckig依赖：

   ```bash
   git submodule update --init --recursive
   GIT_LFS_SKIP_SMUDGE=1 uv sync --locked
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
   # legacy 模式或30000v2 Fast包装：
   uv pip install --python .venv/bin/python -r deploy/fr3_wuji_fast/requirements.txt
   .venv/bin/python -c 'from openpi_client import rtc; import ruckig; print(rtc.PROTOCOL_VERSION, ruckig.__file__)'
   ```

   随后确认 19999 / 54D / JAX 服务配置和设备配置仍指向同一 checkpoint。
2. 先运行快速部署全套无硬件测试：

   ```bash
   env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     .venv/bin/python -m pytest -q -p no:cacheprovider deploy/fr3_wuji_fast
   ```

3. 再运行 RTC 协议、模型引导和动作排列的定向测试：

   ```bash
   env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
     .venv/bin/python -m pytest -q -p no:cacheprovider \
     packages/openpi-client/src/openpi_client/rtc_test.py \
     src/openpi/models/pi0_test.py \
     src/openpi/policies/fr3_wuji_policy_test.py \
     src/openpi/policies/policy_test.py -k rtc
   ```

4. 把 `CKPT` 改为目标机器上完整的19999步骤目录，用完整 RTC 参数做离线检查；该命令不连接模型
   服务：

   ```bash
   CKPT=/absolute/path/to/19999
   bash deploy/fr3_wuji_fast/run.sh --check --checkpoint "$CKPT" \
     --broker-mode rtc \
     --rtc-initial-delay-steps 5 \
     --rtc-delay-buffer-size 10 \
     --rtc-prefix-attention-schedule exp \
     --rtc-max-guidance-weight 5
   ```

   再以同一个 checkpoint 启动 `bash deploy/fr3_wuji_fast/serve.sh --checkpoint "$CKPT"`；只有日志依次
   出现 `Vanilla JAX inference warmup complete`、
   `RTC guided JAX inference warmup complete` 和 `Checkpoint ready`，才表示监听前两条推理签名均已
   编译/预热且 guided 回执已校验。另开终端执行一次不连接设备的合成请求并查看其打印的握手元数据：

   ```bash
   bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/smoke_test.py \
     --uri ws://127.0.0.1:8001 --count 1 \
     --output logs/fr3_wuji/rtc-smoke.json
   ```

   元数据必须包含 `rtc_protocol_version=1`、`rtc_algorithm=soft_prefix_guidance`、
   `rtc_warmed_up=true`，且 `rtc_prefix_attention_schedules` 包含 `exp`。这个 smoke 只发送普通合成推理，
   guided 路径仍以服务启动预热和上一步定向 pytest 为准，输出动作不得送入硬件。
5. 先做 RTG 回归，再在现场值守、低风险姿态和可立即停机条件下做 RTC 实机试验。`--rounds 1`
   没有前一个动作块可供 RTC 拼接，至少需要 `--rounds 2` 才能覆盖 RTC 换块；结束仍建议 `disable`。
   新的默认动态限值模式应与 `--arm-motion-limits legacy` 分开留存命令和日志；在确认模型原始
   相邻节点变化、底层控制器限制与停止链路前，不应直接扩大轮数。
6. 核对 `fast-config.json`、每块 `admission.json` / `inference.npz`、`control/events.jsonl` 和
   `fast-report.json` 中的预测延迟、实测跳步、前缀 horizon、回执绑定和停止结果，再逐步扩大轮数。

2026-09-14 抽查最近20条明确来自19999_269的历史预测，16条通过当时的原始节点限位和机械臂
速度检查，4条超过0.7 rad/s。这是旧阈值下的历史结果，不是当前默认模式的准入结论；它也只验证
静态动作数组，不验证现场首帧、RTG衔接、碰撞、物理跟踪或停止。本次更新没有启动真实机器人。

切换 `--arm-motion-limits` 后需重新运行快速执行命令，模型服务无需重启。

## 底层故障诊断与停止（2026-09-18）

Pi05控制器在首次锁存故障时保留原因：`previous_command_expired`、
`current_command_expired`、`update_gap`、`clock_reversed`、`future_command`、
`invalid_window`、`decode_failed`、`sequence_mismatch` 或 `run_id_mismatch`。
记录故障时刻、原始指令时间和截止时间、预期/实收序号、控制周期间隔；
1kHz实时循环只记录固定大小快照，非实时10ms定时器发布状态并只打印一次详细错误。

左右臂的 `/<side>/joint_impedance_controller/pi05_status` 心跳由设备进程检查。
收到任一底层故障就走原有双臂双手停止路径，故障原因通过IPC返回快速执行端；
网关成功应答不能覆盖已收到的底层故障。启动须具备双臂状态心跳，运行时心跳过期也停止。
停止后继续读取实测位置/速度用于停止确认，不自动恢复发送。

每次运行的 `controller-status.json` 保存首次故障和左右最新状态；`arms.log` 保存
底层首次错误，`fast-report.json` 和 `device-exit.json` 保存上层停止原因。
底层故障回传也保护使用同一Pi05设备桥的慢速部署；快速0.2rad、慢速0.08rad跟踪阈值
及20ms指令有效期均保持各自原值。此修改用于精确定位和及时停止，不放宽时效或序号检查。

修改控制器诊断源后执行（仅生成/构建，不启动设备）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.weight_motion_eval.oneshot.controller_diagnostics
bash experiments/weight_motion_eval/oneshot/build_overlay.sh
```

新建overlay时 `prepare_overlay.py` 会自动加入诊断代码。构建完成后重新运行执行端，
模型服务无需重启。

## Legacy 快速机械臂平滑（2026-09-18；当前非19999默认）

显式选择 `--arm-motion-limits legacy` 时，快速执行端在100Hz下发前使用本地 Ruckig
保留上一帧指令的位置、速度、加速度，
对14个机械臂关节持续生成受约束的目标；动作块切换不重置生成器，块内部也经过同一处理。
RTG 节点拼接与默认 `--guidance-steps 3` 不变；RTC 则直接使用模型 soft prefix guidance 后的动作，
不再做第二次 cubic 拼接。双手目标与SDK控制不变。
首次接近、块间等待和末尾到位均使用同一生成器；末尾必须等生成器到达目标且停止，再检查实测停稳。
急停、时效故障和反馈异常仍直接走原停止流程，不等待平滑。

Ruckig只在legacy模式生效。19999 Fast的RTG/RTC/serial默认不启用它；复用快速执行层的
30000v2包装默认仍选择legacy。RTC只支持19999 / 54D / JAX，不会因此扩展到64维。
中速 `fr3_wuji_medium`、慢速播放器和底层阻抗参数均未修改。此功能是执行端轨迹平滑，
并非模型端RTC。

调参位置：`deploy/fr3_wuji_fast/limits.py`：

```python
ARM_ACCELERATION_RAD_S2 = 3.0  # rad/s²
ARM_JERK_RAD_S3 = 30.0        # rad/s³，加速度变化率
```

legacy速度上限为1.0 rad/s，跟踪阈值在默认和legacy两种模式下都仍为0.2 rad。
降低加速度/jerk一般会使动作柔和，但增大目标跟随延迟；默认值为待现场验证的起点。
这不是整段慢放：模型30Hz时间轴和100Hz下发不变，但机械臂可能滞后于模型目标，
手臂与手指的配合时序、接触动作和成功率需要复测。使用当前采样位置作为静止目标反复重规划，
未将有突变的原始目标速度直接传入。生成器不使用远程服务或中间路点。

Ruckig依赖由legacy路径导入并固定版本；要使用legacy或默认选择legacy的30000v2 Fast包装，
目标部署环境必须安装，不能把当前这台没有可用 `.venv` 的电脑视为已安装：

```bash
.deployment/tools/bin/uv pip install --python .venv/bin/python -r deploy/fr3_wuji_fast/requirements.txt
```

legacy下依赖缺失、版本错误、平滑参数非法会在启动设备前报错。生成失败或预测制动轨迹越界时停止，
不会退回原始目标继续发送。参数保存在 `fast-config.json`；`control/events.jsonl` 的
`arm_smoothing_sample` 以10Hz记录原机械臂目标、生成器加速度与目标偏差，
`frame_submitted.command` 为实际下发位置，`complete` 保存全程最大平滑偏差。
这些平滑字段只应在legacy运行中解释。修改模式或参数后重启快速执行命令即可，模型服务不用重启。
