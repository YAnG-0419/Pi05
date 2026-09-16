# 30000 兼容入口

新入口通过 `bash deploy.sh serve 30000` 启动模型，再选择 `slow --model 30000` 或 `normal --model 30000`。

下面保留原慢速 extended 参数的兼容命令，从仓库根目录运行：

```bash
bash deploy.sh serve slow64 --check-only
bash deploy.sh serve slow64
# 服务就绪后，另一个终端检查：
bash deploy.sh slow64 --check
```

默认权重 `checkpoints/pi05/30000`，服务端口 8002。模型 64 维，设备接口 54 维；输入重排为左臂、右臂、左手、右手并补零，输出反归一化后重排回左臂、左手、右臂、右手。模型输出按绝对关节位置解释。

旧入口调用公共慢速执行器，通过显式 `slow64` 规划配置设置本方案的准入参数。真实执行和依赖见 [根目录 README](../README.md)，[历史实现记录](../docs/history/30000-implementation.md)保留。
