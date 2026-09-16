# DP3 双臂真机部署

这是 `harvesting/deploy` 的独立 DP3 部署版本，复用其 FR3/Wuji 采集与安全检查链路，但拥有独立 Git 历史；向本仓库提交不会修改原 `harvesting/deploy` 项目。

当前交付范围是：真实传感器采集 → DP3 点云/状态预处理 → checkpoint 推理 → 动作安全检查 → 影子日志。**硬件动作输出仍被强制关闭**，完成现场标定、急停和低速验收前不能直接驱动机器人。

## 目标机最短流程

```bash
git clone https://gitlab.litchibot.top/harvesting/dp3_deploy.git
cd dp3_deploy
bash scripts/setup_dp3.sh

export DP3_CHECKPOINT=/绝对路径/latest.ckpt
bash scripts/check_dp3.sh
bash scripts/smoke_dp3.sh
bash scripts/serve_dp3.sh
```

另开终端启动一次真实传感器影子推理：

```bash
cd dp3_deploy
export DP3_CHECKPOINT=/绝对路径/latest.ckpt
export OPENPI_TELEOP_ROOT=/绝对路径/gello_upper_body_teleop
export OPENPI_TELEOP_PYTHON=/绝对路径/gello-upper-body-teleop环境/bin/python
export OPENPI_CAMERA_CONFIG=/绝对路径/cameras.yaml

conda run --no-capture-output -n dp3-deploy \
  env PYTHONPATH="$PWD:$PWD/src:$PWD/packages/openpi-client/src:$PWD/fr3_wuji" \
  python fr3_wuji/client.py \
  --config fr3_wuji/executor.dp3.example.yaml \
  live-shadow --output "logs/dp3-shadow-$(date +%Y%m%d-%H%M%S)" \
  --start-cameras --single-shadow-step
```

checkpoint 不进 Git。默认期望的 SHA256 已写入 `MODEL.sha256`；若训练结果重新导出，需同时更新该文件。

完整的依赖、验收标准和安全边界见 [DP3_DEPLOY.md](DP3_DEPLOY.md)，迁移设计见 [移植.md](移植.md)。
