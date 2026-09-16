# GR00T 工控机单机部署流程

本文只描述一台工控机。Pi05 控制端、GR00T 推理容器、checkpoint、基础模型缓存、
FR3/Wuji 设备桥都位于同一台工控机，不存在第二台推理服务器，也不经过局域网请求
另一台机器。

除最后的 `--execute` 外，本文所有检查都不会向机器人发送动作。

## 0. 固定部署对象

推荐在工控机保持以下目录：

```text
/home/descfly/workspace/Pi05
/home/descfly/workspace/Isaac-GR00T
```

当前候选 checkpoint：

```text
/home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint
```

基础模型缓存：

```text
/home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/models_http
```

已验证的 checkpoint manifest SHA256：

```text
cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

Isaac-GR00T 源码 commit：

```text
3ffdc074df4597599d9e6126e534c6c7474490c9
```

## 1. 单机运行结构

```text
同一台工控机

┌──────────────────────────────────────────────┐
│ Pi05 宿主机 Python 3.11                     │
│ 相机/状态观测 → 时效与限位检查 → 100 Hz 控制 │
└───────────────────┬──────────────────────────┘
                    │ HTTP/NPZ
                    │ 127.0.0.1:8766
┌───────────────────▼──────────────────────────┐
│ GR00T Docker，Python 3.12 + CUDA             │
│ 三路 RGB + 54 维状态 → 16 x 54 动作          │
└──────────────────────────────────────────────┘

Pi05 宿主机 → Unix socket → 现有控制/设备桥容器 → FR3/Wuji
```

GR00T 容器的限制：

- 不挂载 `/dev`。
- 不使用 `privileged`。
- 不获得设备控制 Unix socket。
- checkpoint 和模型缓存均只读挂载。
- 位于 Docker `internal` 网络。
- 只向本机回环地址发布 `127.0.0.1:8766`。
- 不允许其他计算机通过该端口调用模型。

## 2. 工控机：取得部署代码

如果 Pi05 尚未克隆：

```bash
git clone git@github.com:YAnG-0419/Pi05.git /home/descfly/workspace/Pi05
```

已有仓库则更新部署分支：

```bash
cd /home/descfly/workspace/Pi05
git fetch origin
git switch feat/fr3-wuji-groot-deployment
git pull --ff-only origin feat/fr3-wuji-groot-deployment
```

检查当前版本和工作区：

```bash
git branch --show-current
git status --short
git log -1 --oneline
```

工作区应为空，分支应为：

```text
feat/fr3-wuji-groot-deployment
```

## 3. 工控机：检查 Isaac-GR00T 源码

```bash
cd /home/descfly/workspace/Isaac-GR00T
git status --short
git rev-parse HEAD
```

`build_image.sh` 只会将 Isaac-GR00T 已提交的 `HEAD` 放入镜像，并要求该仓库工作区
干净。不要在有未提交代码时强制切换。如果需要切到已验证版本，应先保存自己的修改，
然后执行：

```bash
git switch --detach 3ffdc074df4597599d9e6126e534c6c7474490c9
```

checkpoint 和 `outputs/` 不进入 Git，也不进入 Docker 镜像。

## 4. 工控机：检查 Docker 和 GPU

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

还要确认原 Pi05 硬件控制镜像存在：

```bash
docker image inspect franka-upper-body-teleop:latest >/dev/null
```

如果控制镜像不存在，先按原 Pi05/FR3 部署流程恢复该镜像；不要用 GR00T 推理镜像
代替硬件控制镜像。

## 5. 工控机：建立 Pi05 宿主机环境

Pi05 侧使用项目自己的 Python 3.11 `.venv`，不要激活 Conda：

```bash
cd /home/descfly/workspace/Pi05
bash deploy/fr3_wuji/setup.sh
```

现有启动脚本会清理外部 Conda、ROS 和 Python 环境变量。不要手动 source
`deploy/fr3_wuji/env.sh`。

## 6. 工控机：只读校验 checkpoint 和缓存

```bash
cd /home/descfly/workspace/Pi05
GROOT_ROOT=/home/descfly/workspace/Isaac-GR00T \
bash deploy/fr3_wuji_groot/serve.sh \
  --checkpoint /home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint \
  --check-only
```

确认输出包含：

```text
"model_manifest_sha256": "cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c"
"action_shape": [16, 54]
"reload_verified": true
```

同时应显示两个固定基础模型 revision。该步骤只读取文件并计算哈希，不加载 GPU。

## 7. 工控机：构建 GR00T 基础镜像

检查本机是否已有基础镜像：

```bash
docker image inspect gr00t:latest >/dev/null
```

没有时在同一台工控机构建：

```bash
cd /home/descfly/workspace/Isaac-GR00T
bash docker/build.sh
```

该步骤会下载并安装 CUDA/Python 依赖，耗时和磁盘占用较大，但不会复制 checkpoint。

## 8. 工控机：构建 Pi05 专用推理镜像

```bash
cd /home/descfly/workspace/Pi05
GROOT_ROOT=/home/descfly/workspace/Isaac-GR00T \
GROOT_BASE_IMAGE=gr00t:latest \
GROOT_IMAGE=pi05-groot-server:scratch30k-20260916 \
bash deploy/fr3_wuji_groot/build_image.sh
```

构建脚本通过 `git archive` 复制 Isaac-GR00T 源码，因此不会把 `outputs/`、模型权重、
HF token 或 Git 凭据发送到 Docker build context。

检查镜像及源码版本标签：

```bash
docker image inspect pi05-groot-server:scratch30k-20260916 \
  --format '{{json .Config.Labels}}'
```

## 9. 工控机：建立本机 Compose 配置

```bash
cd /home/descfly/workspace/Pi05
cp deploy/fr3_wuji_groot/.env.example deploy/fr3_wuji_groot/.env.groot
```

确认 `deploy/fr3_wuji_groot/.env.groot` 内容为：

```dotenv
GROOT_IMAGE=pi05-groot-server:scratch30k-20260916
GROOT_CHECKPOINT=/home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint
GROOT_MODEL_CACHE=/home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/models_http
GROOT_PORT=8766
```

`.env.groot` 已被同目录 `.gitignore` 排除。检查 Compose 展开结果：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  config
```

重点确认：

- 两个模型目录均为 `read_only: true`。
- 发布地址是 `127.0.0.1:8766`。
- 没有 `/dev`、`privileged` 或硬件 socket。

## 10. 工控机：容器内只读检查

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  run --rm groot-policy \
  --checkpoint /models/checkpoint \
  --model-cache /models/cache \
  --check-only
```

容器内重新得到的 manifest 必须仍为：

```text
cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

## 11. 工控机：启动 GR00T 模型服务

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  up -d
```

跟随启动日志：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  logs -f groot-policy
```

看到以下内容后，按 `Ctrl-C` 只退出日志跟随：

```text
GR00T_POLICY_READY http://0.0.0.0:8766
```

虽然容器内监听 `0.0.0.0`，Docker 只把它发布到工控机的 `127.0.0.1`，不会对外网卡
开放。

查看健康状态：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  ps
```

## 12. 工控机：模型接口和假观测验收

先校验运行中的模型身份，不连接硬件：

```bash
cd /home/descfly/workspace/Pi05
bash deploy/fr3_wuji_groot/run.sh --server-check \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

再发送三张全零假图像和全零关节状态，只验证模型返回有限 `(16, 54)` 动作：

```bash
bash deploy/fr3_wuji/run.sh \
  -m deploy.fr3_wuji_groot.smoke_client \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

假观测输出只能用于通信验收，禁止发送给硬件。

## 13. 工控机：Pi05 无硬件启动预览

为输出选择一个从未存在过的新目录：

```bash
bash deploy/fr3_wuji_groot/run.sh --check \
  --output /home/descfly/workspace/Pi05/logs/weight_motion_eval/groot-single-host-check
```

该步骤只生成 runtime 和 Docker 命令预览，不启动设备。原 Pi05 路径也必须保持可用：

```bash
bash deploy/fr3_wuji_fast/run.sh --check
```

到这里为止，GR00T 容器可以继续运行，但尚未连接或控制机器人。

## 14. 现场确认后进行单动作块试运行

必须满足：

- 操作人员和物理急停在场。
- 清空机器人工作空间。
- 检查双臂、双手和三路相机状态。
- 确认模型服务健康、SHA256 一致。
- 首轮只运行一个 16 步动作块。

明确决定开始后执行：

```bash
bash deploy/fr3_wuji_groot/run.sh --execute --supervised-trial \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c \
  --rounds 1 \
  --start-cameras \
  --finish-policy disable
```

首轮禁止多轮运行、无人值守或 `finish-policy hold`。

## 15. 日常启动与停止

以后工控机重启后，先启动模型：

```bash
cd /home/descfly/workspace/Pi05
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  up -d
```

等待服务 healthy，再运行 `--server-check`，最后才考虑真机执行。

停止模型服务：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  down
```

## 16. 可选：仅做本机镜像备份

单机部署不需要 `docker save/load`，但可备份已经验证的推理镜像：

```bash
mkdir -p /home/descfly/groot-image-backup
bash deploy/fr3_wuji_groot/export_image.sh \
  pi05-groot-server:scratch30k-20260916 \
  /home/descfly/groot-image-backup/pi05-groot-server-scratch30k-20260916.tar
```

该备份只包含代码和依赖，不包含 checkpoint 或基础模型缓存。

## 故障定位顺序

1. `docker compose config` 失败：检查 `.env.groot` 的本机绝对路径。
2. 容器 `--check-only` 失败：检查 checkpoint 或模型缓存是否完整。
3. 容器 unhealthy：查看 `docker compose logs groot-policy`，检查 GPU、显存和缓存。
4. `--server-check` 失败：检查 `127.0.0.1:8766`、warmup 和模型 SHA256。
5. 假观测失败：保持硬件关闭，先解决模型加载、输出或推理超时。
6. Pi05 `--check` 失败：处理 `.venv`、控制镜像或 overlay，不要尝试 `--execute`。
