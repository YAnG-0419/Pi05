# 30000：64 维模型、54 维 FR3/Wuji 部署接口

默认权重 `checkpoints/30000`，服务地址 `ws://127.0.0.1:8002`。模型维度64、动作块50步，使用 checkpoint 自带 `assets/0909_125/norm_stats.json`。不修改19999服务、训练配置、权重、设备接口或限位。

`checkpoints/30000v2` 是全量微调权重，可通过本目录的独立快速入口运行。`serve_v2.sh` 使用无LoRA分支的 `gemma_2b` 和 `gemma_300m` 参数结构；快速入口只在自身进程内将模型契约替换为64维到54维适配器。原有19999快速入口、30000慢速入口及其LoRA默认结构均不变。

## 数据约定（adapter_revision=2）

外部状态和动作按：左臂7、左手20、右臂7、右手20，单位弧度。

30000 的模型状态和动作按：左臂7、右臂7、左手20、右手20、补零10。机械臂动作按绝对位置解释。该映射依据30000/19999统计对比及deployment-077a549ac7诊断确定；当前checkpoint没有完整训练配置记录。

输入先按模型顺序重排并补零，再用64维统计归一化和tokenize。输出先反归一化，再重排为外部54维顺序，丢弃末10维，不叠加实测关节位置。模型预测的末10维不要求严格为零。

服务与专用执行器检查权重manifest哈希、归一化哈希、模型/物理维度、动作含义、模型顺序及adapter_revision。旧服务必须重启；旧版本元数据会被拒绝，不通过放宽校验接入。

## 使用

在仓库根目录运行：

```bash
bash deploy/fr3_wuji_30000/run.sh --check-only
bash deploy/fr3_wuji_30000/run.sh --smoke-only
bash deploy/fr3_wuji_30000/run.sh
```

`--check-only`仅检查目录和统计。`--smoke-only`加载模型、假观测推理检查有限(50,54)输出后退出。正常服务先预热，再监听8002；Ctrl-C退出。这些模型服务命令不连接机器人。

服务显示就绪后，在另一终端运行专用慢速执行器：

```bash
bash deploy/fr3_wuji_30000/execute.sh \
  --execute --supervised-trial --hand-control slider \
  --continuous --replan-steps 30 --rounds 50 \
  --minimum-time-scale 2.5 \
  --checkpoint /home/user/lpy/Pi05/checkpoints/30000 \
  --uri ws://127.0.0.1:8002 \
  --start-cameras --finish-policy disable
```

该执行命令会启动真实硬件。执行器复用19999当前的慢速规划及设备下发实现，仅在专用入口进程中替换模型契约；不更改旧部署文件。它仍是慢速模式，不是官方RTG快速部署。已有初始偏差、关节限位、速度、反馈和通信保护仍可能拒绝模型预测。

## 30000v2 正常速度部署

先在终端1启动30000v2模型服务：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_30000/serve_v2.sh
```

看到 `Warmup OK: actions=(50, 54)` 和服务监听提示后，在终端2先执行一轮现场测试：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_30000/fast.sh \
  --execute --supervised-trial --start-cameras \
  --rounds 1 --finish-policy disable
```

确认现场行为后，可将 `--rounds 1` 改为需要的轮数，例如 `--rounds 50`。该30000v2 Fast包装
虽然复用19999快速执行层，但**默认显式选择 `--arm-motion-limits legacy`**，不会跟随19999 Fast
新的无动态上限默认值：模型动作30 Hz、控制插值100 Hz，正常回放继续使用旧机械臂1 rad/s、
Ruckig加速度3 rad/s²、jerk 30 rad/s³及网关1.2 rad/s阈值。30000专用的首帧接近准入上限为
1.5 rad，初始动作仍从实测姿态通过受速度、加速度和jerk约束的接近轨迹完成；19999 Fast首帧
接近准入上限仍为0.3 rad。关节位置、0.2 rad跟踪、接触力矩/电流相关保护、20 ms时效、
反馈健康/失联和急停检查均保留。上述是代码默认关系，不是新的真机验证结果。

只做离线配置检查、不连接模型或真机：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_30000/serve_v2.sh --check-only
bash /home/user/lpy/Pi05/deploy/fr3_wuji_30000/fast.sh --check
```

## 验证

离线回归：输入关节逐项映射、补零、完整归一化/tokenize及反归一化链、绝对动作不重复加状态、输出重排、无效数据拒绝、旧服务契约拒绝、旧训练配置不变；真实AsyncRecorder spawn写入/退出通过。

历史失败记录仅作输出解释诊断：撤销旧服务重复加状态并重排后，左臂J4由-3.913变为-1.941rad、J6由6.332变为2.973rad；50×54输出未再发现关节范围越界。这不是重新推理，因为旧记录输入顺序仍错误，且没有保存三路图像，不能用于现场动作。

本次实际模型加载和假观测推理通过：Warmup OK: actions=(50, 54)。假观测通过不代表现场任务已验证；本次没有执行真机动作。

## 30000 规划准入范围

用户要求将此前两项报错阈值一并提高。专用 execute.sh 当前使用：

- 双臂首帧接近偏差 arm_raw_initial_delta_rad：1.5rad（含边界）。
- 双臂平滑节点相对模型原始节点的最大偏差 max_smoothing_delta_rad.arm：0.2rad。

通过当前进程的 planning_profile 实现；旧planner.py、config.yaml和19999入口保持原值0.3rad/0.03rad。手部参数、首步插值、接近和播放的速度/加速度/jerk、硬关节范围、跟踪、接触保护和超时均未改变。两个新阈值保存在生成计划的config中。无需重启模型服务，重新运行专用执行命令即可。

此前0.08rad配置的离线验证：恰好1.5rad可规划，1.500001rad拒绝；大于0.08rad的平滑偏差仍拒绝；其余配置校验保留。三条历史预测（各取前30步、slider、最低慢放2.5倍）完整通过规划：

| 记录 | 首步接近时长 | 播放时长 |
| --- | ---: | ---: |
| e12029f049 | 9.075秒 | 11.632秒 |
| 25015a435c | 9.819秒 | 10.318秒 |
| b349589ad0 | 10.473秒 | 9.537秒 |

以上是历史数据离线规划，不包含现场到位等待，不代表未来预测或真实执行一定通过。本次没有运行真机。

当前0.2rad配置按用户要求更新；手部平滑0.08rad、首帧1.5rad及其他约束不变。a3d935e847第3轮历史预测完整通过离线规划，接近约2.532秒、播放约10.327秒；人工构造的超过0.2rad平滑改动仍被拒绝。未运行真机，模型服务不需要重启。
