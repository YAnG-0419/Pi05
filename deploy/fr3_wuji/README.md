# 本机 FR3/Wuji 推理环境

本目录用于 `/home/user/lpy/Pi05` 的独立部署。训练配置为 `pi05_fr3_wuji`，输入三路 RGB 和 54 维实测状态，输出 `(50, 54)` 绝对关节位置。

模型服务与软件检查入口不会连接硬件。另有独立的[真实观测只读适配入口](OBSERVATIONS.md)，可显式读取设备状态和图像；它不会发送运动指令。**完整真机执行客户端尚未实现，不能仅凭模型服务或观测采集成功就执行机器人任务。**

[真实观测推理记录](INFERENCE_RECORDING.md) 在此基础上请求模型推理，保存动作、关节范围、跳变、延迟和数据时效诊断，不下发动作。

[执行客户端设计](EXECUTOR_DESIGN.md) 定义进程分工、协议、状态机与验收顺序。[影子客户端](EXECUTOR.md) 已实现离线重放、状态机、动作检查、异步日志和现场只读推理入口；[配置](executor.example.yaml) 已由 `client.py` 读取，真实命令发送与连续执行尚未开放。

本机环境、权重加载和模拟推理已通过验收；下方下载接手段落是早期记录，当前环境无需重跑 `finish_setup.sh`。

## 隔离范围

- Python 3.11 虚拟环境：`Pi05/.venv/`。
- uv、独立 Python 解释器和依赖/模型缓存：`Pi05/.deployment/`。
- 检查记录：`Pi05/logs/fr3_wuji/`。
- 原有 `pyproject.toml`、`uv.lock`、模型/训练代码保持不变。
- `/home/user/lpy/gello-retarget` 仅作为只读参考，不修改其源码、Conda、Docker 服务、配置或驱动。
- 使用本机 housekeeping CPU `0-7,16-23`，避开为 Franka 预留的 `8-15`；降低进程优先级，限制 BLAS/OpenMP 为 4 线程，并关闭 JAX 预占显存。仍须在实际联调时测量共享 GPU/内存/CPU 对实时控制的影响。

通过 `run.sh` 启动即可，不必 `conda activate`、`source .venv/bin/activate` 或修改 `.bashrc`。脚本会在子进程中清除外部 Python/ROS/Conda 库路径污染。不要在遥操终端手动 source `env.sh`。

## 环境重建

### 本次下载接手（安装尚未完成）

2026-09-07：Python 3.11、项目隔离入口和 tokenizer 已准备；大依赖下载中断后由用户接手。GPU 运算与完整依赖导入尚未验收。临时下载进程已停止，已完成文件及 `.partial` 断点保存在 `.deployment/wheels/`，清单保存在 `.deployment/dependency-downloads.json`。

查看剩余文件（不会联网）：

```bash
cd /home/user/lpy/Pi05
python3 deploy/fr3_wuji/download_deps.py --status
```

切换到较快的代理节点后续传。以下端口沿用本机此前的 7897；若代理客户端端口不同请替换，仅影响这一次命令：

```bash
https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 \
  python3 deploy/fr3_wuji/download_deps.py
```

脚本按文件显示进度，使用官方锁定 URL 并校验 SHA256，自动跳过已完成文件。`Ctrl-C` 停止后可用同一命令续传。**这一步只下载，不安装，也不启动模型或硬件。** 需要下载器的原始 URL 时见 `.deployment/dependency-downloads.json`。

全部下载完成后，可继续安装和软件验收：

```bash
bash deploy/fr3_wuji/finish_setup.sh
```

该步骤先校验清单中的文件，再安装到 `.venv`，按原 `uv.lock` 做离线同步、检查依赖和 GPU。它尚待下载完成后实际验证；若提示缓存缺包，保留报错继续排查。完成前不要将本环境视为可用的模型服务。

### 从头重建

首次安装 uv（本机本次已完成）：

```bash
cd /home/user/lpy/Pi05
/home/user/miniconda3/bin/python -m pip install \
  --target "$PWD/.deployment/tools" --no-cache-dir 'uv==0.12.10'
```

按仓库锁文件安装，不更新依赖版本：

```bash
bash deploy/fr3_wuji/setup.sh
```

此 FR3 推理路径不需要 ALOHA/LIBERO 的 Git 子模块，也不需要下载训练数据、基础模型 checkpoint、安装 ROS、重装 GPU 驱动或改动现有控制镜像。

## 权重复制前可做的验证

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/doctor.py --prepare-tokenizer
```

检查完整服务导入链、JAX GPU 矩阵运算和 BF16 卷积、模型配置和 tokenizer，写入 `logs/fr3_wuji/environment.json`。第一次加 `--prepare-tokenizer` 会联网下载 tokenizer 到本项目缓存；以后可复用。此检查不加载模型权重。

## 权重回来后

复制完整的训练 checkpoint 步骤目录，例如：

```text
checkpoints/tomato_lora_ep65/19999/
├── params/                         # 整个目录，包含所有参数数据和元数据
└── assets/fr3_wuji/tomato/norm_stats.json
```

不要只复制 LoRA 文件、checkpoint 的父目录或其他实验的归一化文件。推荐将该步骤目录整体复制，复制完成后再启动。若实际步骤不同，替换下面路径。

先做目录与 54 维统计检查：

```bash
bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/serve.py \
  --checkpoint checkpoints/tomato_lora_ep65/19999 --check-only
```

这一步只检查目录和统计结构，不能证明所有参数分片复制完整；后续实际加载才会验证参数兼容性。

在第一个终端启动模型服务：

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/serve.py \
  --checkpoint checkpoints/tomato_lora_ep65/19999
```

默认监听 `127.0.0.1:8000`，固定使用训练任务原文：

```text
Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.
```

在第二个终端做三次假观测推理：

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji/run.sh deploy/fr3_wuji/smoke_test.py
```

检查输出 `(50, 54)` 且全部有限，记录首次编译和后续延迟到 `logs/fr3_wuji/smoke.json`。默认每次响应超时 300 秒，可用 `--timeout` 调整。假观测输出只用于验证模型与通信，不能发给硬件。

服务在前台运行，`Ctrl-C` 停止；未设置开机自启。8000 被占用时，服务用 `--port 8001`，检查端用 `--uri ws://127.0.0.1:8001`。未来确需远程访问时，服务显式加 `--host <本机局域网IP>`。

## 后续真机接入

用户已确认沿用现有遥操的硬件/IP/相机配置。当前参考仓库 commit 为 `efd31cab096e14ff859def70e1d8757191838b91`。

- 现有安全网关允许 `gello/pico/vive/replay`，尚未允许 `openpi`。后续须使用独立的部署配置接入并保留状态新鲜度、接管、速度和限位检查。
- 手部命令遥测话题不是执行入口；SDK 需要独立的唯一拥有者和实测反馈。即使 `auto_enable=False`，构造 Hand 2 后端也可能调用 disable，不能当作无副作用检查。
- 三路 RGB、左右臂和左右手实测状态的只读适配见 [OBSERVATIONS.md](OBSERVATIONS.md)。异步动作执行、超时和停止逻辑仍待实现。
- 观测入口不启动 Docker/ROS 控制栈，不实例化有电机使能/禁用副作用的 Hand 后端，也不做运动测试。

这些模块完成并在用户在场时通过分阶段验证后，才能运行完整真机任务。
