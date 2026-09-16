# 源码来源与适配范围

- `vendor/diffusion_policy/`：来自 `/home/descfly/lcl_work/diffusion_policy/diffusion_policy`，仅复制 `DiffusionUnetImagePolicy` 的推理导入依赖闭包，未复制 workspace、训练入口、数据集或仿真环境。许可证 `vendor/DIFFUSION_POLICY_LICENSE`（MIT）。保留的类可能包含未使用的训练辅助方法，部署不会调用它们。
- `dp_deploy/encoder.py`：来自 `/home/descfly/lcl_work/Convert_data/src/convest/dp/encoder.py`，与 checkpoint 中 `convest.dp.encoder.RGBDObsEncoder` 相同结构。部署中直接构造已知模型类，不执行 checkpoint 的 Hydra target 或 resolver。
- 图像和深度处理：按 `Convert_data/src/convest/dp/convert.py` 的 `resize`、`depth_metres` 及 `dp/dataset.py` 的归一化实现；测试对三个不同源尺寸逐像素比对训练源码结果。
- `dp_deploy/robot/`：来自 `/home/descfly/picking-tomatoes/openpi/experiments/weight_motion_eval/oneshot`。来源 SHA256 和文件清单见 `vendor/sources.json`；上游许可证位于 `vendor/OPENPI_LICENSE`。控制器、手部 SDK、遥操参考保留各自原始许可证。

设备桥复用原始 `DeviceSession`、`ConsumerGuard`、`RemoteDevices`、`HandOwner`、网关/拆分器、反馈与故障处理、20ms 指令信封、期限校验头文件。原协议的 `source=openpi`、`pi05v1` 信封和 ROS 节点名有意保留，以保持末端接口一致；模型与日志使用 DP 身份，未将 DP 伪装成 pi0.5 权重。

设备桥模块的适配仅包括 CPU 诊断子进程的 Python 模块路径。本地 `prepare_overlay.py` 另外适配当前参考控制器版本：从 `<cstdint>` 位置加入 `<atomic>` 和期限头文件，移除该较早版本不存在的 `command_status_` 遥测赋值；保留故障锁存、当前姿态保持、序号和原始期限检查。执行前校验本地编译产物哈希。上游控制器和参考仓库不被修改。

`build.py`、`hardware.py`、`observe.py`、`checkpoint.py`、`control.py`、`client.py` 等是本工程的部署适配。它们替换原脚本中写死的旧主机目录、50步模型和 RGB-only 输入。实际参考控制源码通过配置指定，目前为：

`/home/descfly/picking-tomatoes/teleop/references/Pico_armhand_teleop-main`

配置沿用用户确认的 IP、相机语义；Docker 镜像、SDK 目录、DDS 和 CPU 取自当前机器可用配置。执行控制仍需现场实测，编译和单元测试不构成物理停止验收。
