# 部署代码维护

本目录只维护 FR3/Wuji 推理和部署。外部遥操仓库作为只读依赖，机器人控制保持显式 `--execute`，不可通过放宽设备检查来使测试通过。

在项目根目录进行离线回归：

```bash
JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests fr3_wuji fr3_wuji_fast fr3_wuji_30000 experiments/weight_motion_eval \
  src/openpi/policies/fr3_wuji_policy_test.py src/openpi/transforms_test.py
```

部分现有测试会读取本机遥操仓库的配置和源码，但不连接设备。模型契约测试使用合成权重元数据和测试 tokenizer；GPU 软件检查通过 `fr3_wuji/doctor.py` 单独执行。

已有回归差异及本机缺失依赖见 [检查记录](docs/deployment_status.md)。
