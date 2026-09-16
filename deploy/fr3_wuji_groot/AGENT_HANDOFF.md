# GR00T 真机部署 Agent 交接单

> 状态更新：本文件最初用于交接。Docker、便携缓存和迁移脚本现已实现；实际执行
> 请以 [MIGRATION.md](MIGRATION.md) 为准，下面的历史任务列表仅保留设计依据。

## 一句话目标

在不修改、不破坏现有 Pi05 推理路径的前提下，把 GR00T 做成可迁移的 GPU
推理容器；真机上位机继续用 Pi05 原生 Python 3.11 环境负责观测、动作准入和
100 Hz 硬件控制，二者通过本机 HTTP 通信。

## 当前状态（2026-09-16）

- Pi05 仓库：`/home/descfly/workspace/Pi05`
- 当前分支：`feat/fr3-wuji-groot-deployment`
- 分支起点：`0b170fb70cbf114869d7a52ac8566dd31f052c1a`
- 所有新增文件都在 `deploy/fr3_wuji_groot/`，目前仍是未提交文件。
- 原 `deploy/fr3_wuji/`、`deploy/fr3_wuji_fast/` 和其他 tracked 文件修改数为 0。
- Isaac-GR00T 仓库：`/home/descfly/workspace/Isaac-GR00T`
- 当前 Isaac-GR00T commit：`3ffdc074df4597599d9e6126e534c6c7474490c9`
- 当前开发机没有构建好的 `gr00t`、`pi05-groot-server` 或
  `franka-upper-body-teleop` Docker 镜像。
- 当前 Pi05 仓库没有 `.venv`，所以尚未在本机执行真实 Pi05 启动脚本。
- 已完成 22 项无硬件 CPU 测试，Ruff、格式、Shell 语法和 Compose 配置检查通过。
- 没有加载 GPU 模型、连接设备或执行机械臂。

## 已完成的代码

- `model_server.py`：GR00T Python 3.12 模型服务，提供 `/healthz`、
  `/metadata`、`/action`。
- `client.py`：Pi05 Python 3.11 HTTP/NPZ 客户端。
- `contract.py`：固定 `16 x 54`、30 Hz、绝对关节目标协议及关节顺序转换。
- `consumer.py`、`timeline.py`、`control.py`：复用原 Pi05 限位、动作准入、
  `ConsumerGuard` 和 100 Hz 控制边界，首版仅支持串行动作块。
- `deploy.py`：默认 `--check`；真机必须同时提供 `--execute`、模型 SHA256，
  以及 `--supervised-trial` 或 qualification。
- `serve.sh`、`run.sh` 和 `README.md`：原生环境启动入口和操作说明。

现有 Pi05 命令必须继续保持不变，例如：

```bash
bash deploy/fr3_wuji_fast/run.sh --check
```

## 已验证的模型

Checkpoint：

```text
/home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint
```

大小约 8.9 GB，完整 manifest SHA256：

```text
cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

此外运行时依赖约 12 GB 的基础模型缓存：

```text
/home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/models_http
```

Checkpoint 和基础模型缓存均不得提交普通 Git，也不应打进最终镜像；应作为只读
volume 单独迁移和挂载。

## 最终进程拓扑（必须遵守）

```text
FR3 / Wuji / 三路相机
          |
          | ROS、SDK、Unix socket
          v
Pi05 宿主机 Python 3.11
观测 -> HTTP/NPZ -> GR00T GPU Docker（Python 3.12）
     <- 16x54 动作 -|
限位/时效/模型指纹准入 -> 独立 100 Hz 控制 -> 现有设备桥
```

- GR00T 容器只允许 GPU、模型只读卷和 TCP 端口，不得获得机械臂、Wuji、
  `/dev` 或控制 socket 权限。
- 实时控制不得移入 GR00T 容器。
- 同机部署使用 Docker `internal` 网络；容器内监听 `0.0.0.0:8766`，仅映射
  `127.0.0.1:8766:8766`，Pi05 使用 `http://127.0.0.1:8766`。
- 当前 HTTP 服务无认证，不得暴露到公网。

## 下一位 Agent 的明确任务

### 1. 让模型缓存路径可迁移

当前 `model_server.py` 从 Isaac-GR00T 的 `factorial_experiment.CACHE` 读取固定缓存，
而 `models_http/models.json` 的 `snapshot` 字段还记录了开发机绝对路径。容器换路径后
会失败。

要求：

- 给模型服务增加显式 `--model-cache` 参数。
- 不修改 Isaac-GR00T 仓库；在 Pi05 新部署模块内完成便携式缓存校验/配置。
- 依据 `models.json` 中记录的 model ID 和 revision，从挂载目录的
  `hub/models--*/snapshots/<revision>` 重新解析实际 snapshot，不信任旧绝对路径。
- 继续设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，禁止运行时偷偷联网。
- 添加临时目录单元测试，覆盖“旧绝对路径无效但 revision 对应本地 snapshot 有效”。

### 2. 新增专用推理镜像

建议新增：

```text
deploy/fr3_wuji_groot/Dockerfile.inference
deploy/fr3_wuji_groot/compose.groot.yaml
deploy/fr3_wuji_groot/build_image.sh
deploy/fr3_wuji_groot/export_image.sh
deploy/fr3_wuji_groot/.env.example
```

要求：

- 基于 Isaac-GR00T 的 `gr00t` 依赖镜像（其 venv 位于 `/opt/gr00t-venv`）。
- 上游 `docker/Dockerfile` 只含依赖、不含源码；专用镜像必须包含固定 commit 的
  Isaac-GR00T 源码和 Pi05 的 `deploy/fr3_wuji_groot` 源码。
- 可使用 BuildKit named additional context 引入 sibling Isaac-GR00T 仓库，不能把
  其 20 GB `outputs/` 复制进 build context/image。
- 为镜像写入 Pi05 commit、Isaac-GR00T commit 和构建时间 label。
- 容器入口只启动 `deploy.fr3_wuji_groot.model_server`。
- checkpoint 挂载到 `/models/checkpoint:ro`。
- 基础模型缓存挂载到 `/models/cache:ro`。
- 使用 `--gpus all`、`--ipc=host`；不得使用 `--privileged`，不得挂载 `/dev`。
- 增加 `/healthz` healthcheck；注意服务只有模型加载和 warmup 完成后才应视为 ready，
  必要时将 liveness/readiness 语义拆清楚。
- 不在镜像中复制私钥、HF token、Git 凭据或用户缓存。

### 3. 提供迁移/导出方式

`export_image.sh` 至少应支持：

```text
docker save pi05-groot-server:<tag> -> tar（可选 zstd）
生成镜像 tar 的 SHA256 文件
```

README 需要同时说明：

- 有 registry：`docker push` / `docker pull`。
- 无 registry：`docker save` / 复制 / `docker load`。
- Git 只迁移代码；checkpoint 和基础模型缓存用 rsync/scp/NAS 单独迁移。
- 在目标机重新运行 checkpoint `--check-only`，确认 manifest SHA256 仍是
  `cf9fd5...66889c`。

### 4. 真机侧环境说明

- Pi05 控制侧不使用 Conda，按照现有仓库创建 Python 3.11 `.venv`：

  ```bash
  bash deploy/fr3_wuji/setup.sh
  ```

- 真机还必须已有或单独迁移 `franka-upper-body-teleop:latest` 控制镜像。
- 目标机需要 NVIDIA driver、Docker 和 NVIDIA Container Toolkit。
- 先执行模型 `--check-only`、容器健康检查、Pi05 `--check`、`--server-check`；
  未完成这些步骤前不得使用 `--execute`。

### 5. Git 收口

- 保留当前功能分支，不要把新增文件直接写到 `main`。
- 先检查 `git status`，不得覆盖用户已有修改。
- 所有修改应继续局限在 `deploy/fr3_wuji_groot/`；如确需修改其他文件，先说明理由。
- 完成并通过测试后再提交和推送当前功能分支；不要提交权重、缓存、日志、`.venv`
  或 Docker tar。

## 验收标准

软件验收（不得连接硬件）：

1. 原有 Pi05 tracked 文件保持不变，原 Pi05 命令和参数不变。
2. 现有测试继续通过，并包含缓存便携性和 Docker 配置测试。
3. Ruff、format、`bash -n`、`docker compose config` 通过。
4. 镜像中能导入 `TomatoPolicy`，能读取只读 checkpoint/cache。
5. `--check-only` 输出的模型 manifest 等于上面的固定 SHA256。
6. 容器 warmup 后 `/metadata` 显示 `warmed_up: true`。
7. 宿主机执行 `run.sh --server-check --expected-model-sha256 ...` 成功，且不接触设备。
8. 通过假观测完成一次 HTTP 请求，返回有限的 `(16, 54)` 动作。
9. 镜像检查确认没有 checkpoint、基础模型权重、凭据和设备权限。

真机执行不属于默认自动验收。任何 `--execute` 都必须由用户现场明确决定，并准备
物理急停、空场、低风险初始姿态和 `--supervised-trial --rounds 1`。

## 当前测试命令

开发机暂时借用 Isaac-GR00T venv 执行无硬件测试：

```bash
cd /home/descfly/workspace/Pi05
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/descfly/workspace/Isaac-GR00T/.venv/bin/python \
  -m pytest -q -p no:cacheprovider deploy/fr3_wuji_groot

/home/descfly/workspace/Isaac-GR00T/.venv/bin/python \
  -m ruff check deploy/fr3_wuji_groot

/home/descfly/workspace/Isaac-GR00T/.venv/bin/python \
  -m ruff format --check deploy/fr3_wuji_groot

bash -n deploy/fr3_wuji_groot/run.sh deploy/fr3_wuji_groot/serve.sh
git diff --check
```

## 非目标

- 不重构整个 Pi05 仓库。
- 不把 Pi05 和 GR00T 合并到同一个 Python 环境。
- 不把 100 Hz 控制移到模型容器。
- 不把模型权重打进 Git 或最终 Docker 镜像。
- 不在开发机自动启动真机执行。
