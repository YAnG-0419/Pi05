# 本机验证记录

日期：2026-09-16。所有本次验证均未连接机械臂或手，未启动真实控制器、未使能或发送真机动作。

| 验证 | 结果 / 证据 |
|---|---|
| 权重完整性 | SHA256 `29639f2df6b50a5ed15e746ecd6de6fd13b9e3435f57ff72ac288a92a42c98b8`，与原目录 `SHA256SUMS.txt` 一致 |
| 权重读取 | `torch.load(weights_only=True, mmap=True)`，仅允许必要配置类型；无通用 dill/pickle 执行，无训练 workspace 初始化 |
| 真实 EMA 模型 | RTX 5090，输出 8×54、完整预测 16×54，有限值；严格加载全部 state dict 和 normalizer |
| GPU 冒烟延迟 | 0.723 / 0.451 / 0.451 秒；`reports/model-smoke.json` |
| 真实数据回放 | 本地 episode60_30hz，第100、101帧，三路 RGB + 头部深度 + 实测关节；HTTP 请求约 0.467 秒 |
| 真实预测接管检查 | 最大首步臂差 0.0733 rad、手差 0.4244 rad；通过限位和初始距离检查；无手部投影；受约束计划约2.49秒 |
| 原控制器隔离副本 | Docker 无网络、无设备挂载构建成功；`reports/build.log`、`.deployment/overlay/source-manifest.json` |
| ROS/SDK 依赖 | 禁网无设备容器导入通过；`reports/doctor.json` |
| ROS 观测链路 | 独立网络命名空间、独立 DDS domain 219，虚拟 sensor_msgs 发布 → 实际采集进程 → HTTP 取出两帧54维/RGBD；`reports/ros-smoke/report.json` |
| 自动测试（65项全部通过） | 关节顺序、按名称重排、RGB逐像素对齐训练、深度单位/无效值、ROS BGR/步长/字节序、因果历史/过期/时钟回退、轨迹速度/限位、实际 ConsumerGuard、有限执行、推理失败停止、IPC失效、手部故障/稳定等待、设备 watchdog、C++ 期限检查。以 `reports/tests.xml` / `reports/tests.log` 为准 |

未验证：实际 FCI/手部设备连接、相机现场映射与深度开启、ROS 现场发现、实时控制调度、真机跟踪/接触与停止、番茄采摘任务效果。当前执行器为受约束逐块验证，未声称实现 30 Hz 原速连续策略播放。
