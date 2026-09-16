# 19999 慢速执行与公共设备层

从仓库根目录运行 `bash deploy.sh serve slow54` 启动模型服务，另一个终端运行 `bash deploy.sh slow54 --check` 检查执行依赖。

本模块保留慢速规划、连续分段推理、设备独占、双臂安全网关、Wuji SDK 桥接和控制器期限补丁。原速和 30000 执行器也复用这些设备模块。

当前启动流程见 [根目录 README](../../../README.md)，过去的开发与测试记录见 [历史记录](../../../docs/history/19999-implementation.md)。
