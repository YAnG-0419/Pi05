# 真实观测适配（只读）

入口是 `observe.py`。它收集、检查并保存模型输入，不加载策略、不请求推理，也没有动作输出接口。模型服务、原遥操环境和代码不作修改。

2026-09-07 本机验收：`direct + sdk + --start-cameras` 预热 35 秒后，30 秒得到 592 组有效观测，组内最大时间差 28.1 ms，选中样本最大年龄 181.1 ms；7 次组装尝试因过期被拒绝。原模型输入预处理和 7 项离线测试通过。报告在 `logs/fr3_wuji/observations-real-03/`。默认 ROS 状态订阅路径已实现，尚未在运行中的遥操控制栈上做本轮联合验收。

## 启动

在已经有 ROS 状态和相机话题时，仅订阅：

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/observe.py \
  --duration 30 --output logs/fr3_wuji/observations-session
```

ROS 路径需要现有进程持续提供 `/left/franka/joint_states`、`/right/franka/joint_states`、`/teleop/wuji/{left,right}/joint_states` 和三路相机图像。这条命令不会启动这些状态源。手部 command 话题从不作为反馈来源。

没有遥操或其他硬件所有者时，可以显式使用本机只读采集：

```bash
bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/observe.py \
  --arm-source direct --hand-source sdk --start-cameras \
  --duration 30 --output logs/fr3_wuji/observations-direct-session
```

每次选择新的输出目录，已有记录不会覆盖。`Ctrl-C` 停止本次采集并关闭本次创建的进程。

`observe.py` 默认仍使用原采集相机配置。只为模型采集 RGB 时，可添加 `--camera-profile rgb`，在输出目录生成本次专用配置，关闭深度、SDK 帧同步和帧聚合；三路颜色输入及其源时间戳契约保持一致。该参数只能与 `--start-cameras` 一起使用，不会调整已有相机进程。推理记录入口的优化默认值见 [真实推理记录说明](INFERENCE_RECORDING.md)。

- `direct` 双臂使用 `libfranka::Robot::read`，以 100 Hz 转发读取结果；不调用 control、错误恢复或参数设置。
- `sdk` 双手直接连接既有 Conda 环境中的 Wuji SDK，验证左右身份和 20 个在线关节，只读取/订阅，不实例化会使能或禁用电机的遥操后端。
- `--start-cameras` 只启动原有三相机入口，预热至少 35 秒，之后仍必须通过真实数据检查。
- 已有遥操控制器、手部 SDK 所有者或 Viewer 时，不要另开直接连接；使用现有只读话题。入口会检查常见占用，但不能识别任意外部程序或防止其他程序在检查后抢占设备。
- 默认使用本机既有 Humble 镜像、ROS domain 0 和 Cyclone DDS。本机硬件身份来自只读参考配置；这不是跨机器自动发现工具。

## 输入契约

| 模型键 | 内容 |
| --- | --- |
| `observation/state` | float32，54 维，`[左臂7、左手20、右臂7、右手20]`，实测位置，弧度 |
| `observation/image` | cam0 / head，RGB uint8，400×640×3 |
| `observation/left_wrist_image` | cam1 / left_wrist，RGB uint8，480×640×3 |
| `observation/right_wrist_image` | cam2 / right_wrist，RGB uint8，480×640×3 |
| `prompt` | 与番茄任务训练一致的英文指令 |

启动时核对原 `dataset_contract.yaml` 的单位和关节名称，以及 `cameras.yaml` 的语义与序列号。关节按名字重排；SDK 手部以 node ID 转为训练所用固件顺序，不能假定收到的数组已有正确次序。缺关节、重名、跨左右手的名字及 NaN/Inf 会被拒绝。

图像通过 `cv_bridge` 明确转换为 `rgb8`，保留原始尺寸。客户端不做归一化、缩放、补边或动作 delta 转换；这些仍由原模型服务的训练配置处理。54 维状态经测试与训练 108 维状态切片结果一致。

## 时间匹配与限制

每台相机用独立 ROS 接收进程，图像订阅采用 `RELIABLE / KEEP_LAST(10) / VOLATILE`，与原 `record_gello.yaml` 的采集设置一致。该配置已注明本机 Cyclone DDS 的 BEST_EFFORT 大图像接收会严重丢帧；仅延长预热不能替代正确的订阅方式。图像、机械臂和手状态通过本机 Unix socket 进入有界缓存。缺失和无效数据不会被零填充。

- 以各路最新图像中最早的时间戳为参考，在各路短历史中选择时间最近的样本。
- 整组样本时间跨度最多 80 ms；选中样本最多距现在 200 ms。
- 双臂、双手的最新反馈还必须在 150 ms 内持续更新。
- 同时检查源时间戳和本机接收时间。延迟到达的旧帧不会因为刚收到而被当作新帧。
- 重复/倒退的时间戳、过期数据或主机时钟跳变会使对应输入失效；不放宽限制凑齐观测。

时间匹配是软件近似对齐，**不是相机曝光的硬件同步**。图像使用原全局 ROS 时间戳，SDK 手状态使用 SDK 对齐的主机时钟。FR3 直接读取的设备时间是启动后的计时，不能与 Unix 时间直接比较，因此使用读取回调的主机时间近似；网络传输误差仍存在。ROS 模式使用已有状态消息的 header 时间戳。

模型训练的 20/30/30 Hz 图像不会产生严格同步的三路 30 Hz 新帧，因此组合观测速率与动作执行速率是两回事。本模块不实现动作调度。

## 记录

输出目录包含 `first.npz`、`last.npz`（真实观测）、`report.json`（选中时间、各路年龄、时间跨度及拒绝计数）和各采集端日志。NPZ 可用 `np.load(path, allow_pickle=False)` 打开，读取后将 `prompt` 转为 Python 字符串。

`captured` 仅表示成功收集了观测，不是动作执行许可或长时间稳定性保证。每个观测都必须在实际使用时再次检查年龄，磁盘保存的观测仅用于离线验证，不能作为实时执行输入。

原遥操仓库始终只读挂载。新增 C++ 读取器仅编译到 `Pi05/.deployment/observation-build/`，不安装到系统，也不修改原镜像、驱动、CPU/IRQ 配置。

## 离线测试

```bash
env -u PYTHONPATH -u PYTHONHOME \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest deploy/fr3_wuji/test_observation.py -q
```

包括关节契约、训练状态切片一致性、RGB 通道、历史状态选择、缺失/过期/时钟异常，以及本地二进制图像传输。测试不连接硬件；本地套接字测试需要运行环境允许 Unix socket 通信。
