# 公共部署环境与观测模块

机器路径在 `paths.py`，环境隔离在 `env.sh`。当前安装、检查和启动流程见 [根目录 README](../README.md)。

- `bash ../deploy.sh check`：本机依赖检查，不连接设备。
- `run.sh`：使用本目录所属项目的虚拟环境与源码运行 Python。
- `observe.py`、`record_inference.py`：真实观测和推理记录。
- `doctor.py`：GPU 运算、推理导入链和 tokenizer 检查。
- `executor/`：保留影子执行和离线诊断工具。

三套真机执行入口由根目录 `deploy.sh` 提供。旧 EXECUTOR/OBSERVATIONS 文档记录具体实现，机器路径和当前启动方法以根目录 README 为准。
