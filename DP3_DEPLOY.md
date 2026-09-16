# DP3 双臂真机侧部署

当前实现完成了真实传感器读取、DP3 点云预处理、模型推理和安全影子评估；**不会向机械臂或灵巧手发送命令**。先在目标机完成本页验证，再单独做有人值守的硬件执行验收。

## 已固定的模型契约

- 输入：最近 4 帧 `point_cloud (1024,3)` 与 `agent_pos (54,)`，均为 `float32`。
- 状态/动作顺序：左臂 7、左手 20、右臂 7、右手 20。
- 输出：8×54 绝对关节位置，20 Hz；不是 delta action。
- 点云：cam0 深度光学坐标系、米、5 mm voxel、确定性 FPS；裁剪值见 `configs/crop_config.json`。
- checkpoint SHA256：由 `MODEL.sha256` 强制校验。

## 1. 目标机准备

目标机需要 NVIDIA 驱动、CUDA 12.8 兼容 GPU、Conda、Docker、现有遥操仓库和控制镜像。克隆本仓库后：

```bash
git clone https://gitlab.litchibot.top/harvesting/dp3_deploy.git
cd dp3_deploy
bash scripts/setup_dp3.sh
export DP3_CHECKPOINT=/安全传输位置/latest.ckpt
bash scripts/check_dp3.sh
```

`dp3_runtime/` 已包含 checkpoint 加载所需的 DP3 源码，不需要训练数据或完整训练仓库。checkpoint 不进 Git，应单独传输；不设置 `DP3_CHECKPOINT` 时默认读取 `models/latest.ckpt`。

目标机必须按实际安装位置设置真机依赖，不能照搬开发机绝对路径：

```bash
export OPENPI_TELEOP_ROOT=/绝对路径/gello_upper_body_teleop
export OPENPI_TELEOP_PYTHON=/绝对路径/gello-upper-body-teleop环境/bin/python
export OPENPI_CAMERA_CONFIG=/绝对路径/cameras.yaml
# 镜像名不是默认值时再设置：
export OPENPI_CONTROL_IMAGE=franka-upper-body-teleop:latest
```

CUDA 不是 12.8 时，先按目标机驱动选择 PyTorch 官方 wheel，再调整 `scripts/setup_dp3.sh` 中两个版本号和下载源；不要让 PyTorch 静默退回 CPU。

## 2. 无机器人验证

```bash
bash scripts/smoke_dp3.sh       # 加载真实权重并用假输入跑一次，不连接硬件
bash scripts/serve_dp3.sh       # 默认 ws://127.0.0.1:8001
```

`smoke_dp3.sh` 成功标准是打印 `DP3 warmup OK: (8, 54) on cuda:0`。

若服务和机器人采集不在同一台电脑：

```bash
DP3_HOST=0.0.0.0 bash scripts/serve_dp3.sh
```

并将 `fr3_wuji/executor.dp3.example.yaml` 的 `model.uri` 改为模型机 IP。只在可信内网开放端口；该协议没有身份认证。

## 3. 真实传感器影子推理

先按现有真机部署填写相机序列号和路径变量，确保没有别的进程占用相机或设备。模型服务保持运行，在另一终端执行：

```bash
conda run --no-capture-output -n dp3-deploy \
  env PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src:$PWD/fr3_wuji" \
  python fr3_wuji/client.py \
  --config fr3_wuji/executor.dp3.example.yaml \
  live-shadow --output "logs/dp3-shadow-$(date +%Y%m%d-%H%M%S)" \
  --start-cameras --single-shadow-step
```

这会启动现有 ROS Docker 采集进程，读取 cam0 深度及双臂双手实测状态，累积 4 帧后请求 DP3，并把动作送入限位、首帧偏差、速度和时效检查。输出只写日志，`hardware_output` 始终为 `false`。

检查输出目录中的：

- `client-report.json`：推理是否完成、是否因时效/限位被拒绝；
- `events.jsonl`：端到端延迟和安全检查原因；
- `artifacts/*.npz`：4 帧点云、状态及 8×54 预测动作；
- `ros-producer.txt`、`cam0-reader.txt`：相机帧、坐标系或点云处理错误。

## 4. 上真机动作前仍需完成

当前执行器刻意没有硬件输出入口。首次动作测试前至少需要现场确认 cam0 坐标系和裁剪可视化、关节零位/顺序、DP3 首帧偏差、20 Hz 时间语义、急停、双手设备独占，以及控制器最终消费端 deadline。不要通过改 YAML 中的 `hardware_output` 绕过它；配置加载器会拒绝。

建议提交内容：本仓库代码、`MODEL.sha256`、影子运行日志和 checkpoint 的安全传输位置。不要提交 checkpoint、训练数据、`.conda` 或设备密钥。
