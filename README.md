# DP3 双臂真机部署

这是 `harvesting/deploy` 的独立 DP3 部署版本，复用其 FR3/Wuji 采集与安全检查链路，但拥有独立 Git 历史；向本仓库提交不会修改原 `harvesting/deploy` 项目。

已接通：真实 cam0 深度 + 54 维双臂双手状态 → 4 帧 DP3 观测 → checkpoint 推理 →
`8×54 @ 20 Hz` 安全规划 → 100 Hz 插值 → 原 deploy 的双 FR3/双 Wuji 硬件桥。
默认不输出动作；真机必须显式 `--execute`，并通过现场急停、只读和低速验收。

## 目标机最短流程

```bash
git clone https://gitlab.litchibot.top/harvesting/dp3_deploy.git
cd dp3_deploy
bash scripts/setup_dp3.sh

export OPENPI_TELEOP_ROOT=/绝对路径/gello_upper_body_teleop
export OPENPI_TELEOP_PYTHON=/绝对路径/gello-upper-body-teleop环境/bin/python
export OPENPI_CONTROL_IMAGE=franka-upper-body-teleop:latest
bash deploy.sh prepare-controller
bash deploy.sh build-controller

export DP3_CHECKPOINT=/绝对路径/latest.ckpt
bash scripts/check_dp3.sh
bash scripts/smoke_dp3.sh
bash scripts/serve_dp3.sh
```

另开终端启动一次真实传感器影子推理：

```bash
cd dp3_deploy
cp fr3_wuji/executor.dp3.example.yaml fr3_wuji/executor.dp3.local.yaml
export DP3_CHECKPOINT=/绝对路径/latest.ckpt
export OPENPI_TELEOP_ROOT=/绝对路径/gello_upper_body_teleop
export OPENPI_TELEOP_PYTHON=/绝对路径/gello-upper-body-teleop环境/bin/python
export OPENPI_CAMERA_CONFIG=/绝对路径/cameras.yaml

conda run --no-capture-output -n dp3-deploy \
  env PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src:$PWD/fr3_wuji" \
  python fr3_wuji/client.py \
  --config fr3_wuji/executor.dp3.local.yaml \
  live-shadow --output "logs/dp3-shadow-$(date +%Y%m%d-%H%M%S)" \
  --start-cameras --single-shadow-step
```

checkpoint 不进 Git。默认期望的 SHA256 已写入 `MODEL.sha256`；若训练结果重新导出，需同时更新该文件。

硬件分级检查入口：

```bash
DP3_CHECKPOINT=/绝对路径/latest.ckpt bash scripts/hardware_test_dp3.sh check
DP3_CHECKPOINT=/绝对路径/latest.ckpt bash scripts/hardware_test_dp3.sh read-only
```

第三阶段会真实运动机器人，不应从 README 直接复制执行。完整依赖、点云契约、急停前置条件、
监督低速试运行和正式启动命令见 [部署.md](部署.md)，迁移设计见 [移植.md](移植.md)。
