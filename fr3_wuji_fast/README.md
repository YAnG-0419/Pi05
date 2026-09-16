# 19999 原速部署

从仓库根目录运行：

```bash
bash deploy.sh serve slow54
# 服务就绪后，另一个终端检查：
bash deploy.sh normal --check
```

模型时间轴 30 Hz、设备下发 100 Hz，支持 RTG 异步推理。模型及动作接口为 50×54。默认权重 `checkpoints/pi05/19999`，服务端口 8001。

真实执行参数、外部设备依赖见 [根目录 README](../README.md)。[上游来源](UPSTREAM.md)及[历史实现记录](../docs/history/19999-fast-implementation.md)保留。
