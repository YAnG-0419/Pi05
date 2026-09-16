# GR00T 从远端开发机迁移到真机上位机

本文使用当前候选 checkpoint：

```text
newdata_v1_scratch30k_20260916/best_checkpoint
```

部署校验指纹：

```text
cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

流程默认使用一台 Linux 真机上位机同时运行 Pi05 控制端和 GR00T GPU 容器。
所有 `--execute` 之前的步骤都不会向机器人发送动作。

## 0. 迁移内容

分别迁移四类内容，不要混在一起：

| 内容 | 方式 | 是否进入 Git/推理镜像 |
|---|---|---|
| Pi05 新部署代码 | Git 分支 | Git，是 |
| Isaac-GR00T 源码 | 构建时写入推理镜像 | 镜像，是 |
| GR00T checkpoint，约 8.9 GB | `rsync`/NAS | 否，只读挂载 |
| 基础模型缓存，约 12 GB | `rsync`/NAS | 否，只读挂载 |
| 推理镜像 | Registry 或 `docker save/load` | 独立镜像包 |
| 原硬件控制镜像 | 真机已有或单独迁移 | 独立镜像包 |

最终运行关系：

```text
Pi05 宿主机进程 <--HTTP 127.0.0.1:8766--> GR00T Docker
Pi05 宿主机进程 --Unix socket--> 现有控制/设备桥容器 --> FR3/Wuji
```

GR00T 容器没有 `/dev`、机械臂网络配置或设备控制 socket。容器位于 Docker
`internal` 网络，只把 `8766` 映射到宿主机 `127.0.0.1`。

## 1. 开发机：固定并推送代码

先确认当前分支，检查待提交文件：

```bash
cd /home/descfly/workspace/Pi05
git branch --show-current
git status --short
```

应处于：

```text
feat/fr3-wuji-groot-deployment
```

测试通过后提交并推送：

```bash
git add deploy/fr3_wuji_groot
git commit -m "feat: add isolated GR00T FR3 Wuji deployment"
git push -u origin feat/fr3-wuji-groot-deployment
```

不要提交 `outputs/`、权重、模型缓存、`.venv`、日志或导出的 Docker tar。

## 2. 开发机：再次只读校验候选权重

```bash
cd /home/descfly/workspace/Pi05
GROOT_ROOT=/home/descfly/workspace/Isaac-GR00T \
bash deploy/fr3_wuji_groot/serve.sh \
  --checkpoint /home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint \
  --check-only
```

确认输出同时包含：

```text
"model_manifest_sha256": "cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c"
"output_shape": [16, 54]
"reload_verified": true
```

该步骤会读取并哈希权重，不加载 GPU。

## 3. 开发机：构建基础 GR00T 镜像

当前专用镜像以官方仓库的 `gr00t:latest` 依赖镜像为基础。若本机还没有该镜像：

```bash
cd /home/descfly/workspace/Isaac-GR00T
git status --short
git rev-parse HEAD
bash docker/build.sh
```

本次适配验证过的 Isaac-GR00T commit 是：

```text
3ffdc074df4597599d9e6126e534c6c7474490c9
```

基础镜像构建需要网络、磁盘空间和较长时间。构建后检查：

```bash
docker image inspect gr00t:latest >/dev/null
```

如果开发机与真机 CPU 架构不同，不要导出开发机镜像；应在目标架构机器上重新构建。

## 4. 开发机：构建专用推理镜像

```bash
cd /home/descfly/workspace/Pi05
GROOT_ROOT=/home/descfly/workspace/Isaac-GR00T \
GROOT_BASE_IMAGE=gr00t:latest \
GROOT_IMAGE=pi05-groot-server:scratch30k-20260916 \
bash deploy/fr3_wuji_groot/build_image.sh
```

构建脚本使用 `git archive` 提取 Isaac-GR00T 的已提交源码，避免把 `outputs/`、
本地凭据和 20 GB 模型文件发送到 Docker build context。专用镜像同样不包含权重。

检查镜像没有设备权限或模型挂载：

```bash
docker image inspect pi05-groot-server:scratch30k-20260916 \
  --format '{{json .Config.Labels}}'
```

## 5. 开发机：配置并进行容器软件验收

创建本机配置；该文件仅保存路径，不提交：

```bash
cd /home/descfly/workspace/Pi05
cp deploy/fr3_wuji_groot/.env.example deploy/fr3_wuji_groot/.env.groot
```

检查 `deploy/fr3_wuji_groot/.env.groot` 中两个绝对路径存在，然后验证 Compose：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  config --quiet
```

先在容器里只读检查 checkpoint 和可迁移基础模型缓存：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  run --rm groot-policy \
  --checkpoint /models/checkpoint \
  --model-cache /models/cache \
  --check-only
```

然后启动模型并等待 GPU warmup：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  up -d

docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  logs -f groot-policy
```

看到以下日志后按 `Ctrl-C` 退出日志跟随，不会停止容器：

```text
GR00T_POLICY_READY http://127.0.0.1:8766
```

## 6. 开发机：验证服务和一次假观测推理

以下命令不连接硬件：

```bash
bash deploy/fr3_wuji_groot/run.sh --server-check \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

再发送三张全零假图像和全零关节状态，只检查真实模型能返回有限的 `(16, 54)`：

```bash
bash deploy/fr3_wuji/run.sh \
  -m deploy.fr3_wuji_groot.smoke_client \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

假观测输出禁止发送给硬件。

停止模型服务：

```bash
docker compose \
  --env-file deploy/fr3_wuji_groot/.env.groot \
  -f deploy/fr3_wuji_groot/compose.groot.yaml \
  down
```

## 7. 开发机：导出推理镜像

准备一个新的输出目录，再导出镜像及 SHA256：

```bash
mkdir -p /home/descfly/groot-transfer
bash deploy/fr3_wuji_groot/export_image.sh \
  pi05-groot-server:scratch30k-20260916 \
  /home/descfly/groot-transfer/pi05-groot-server-scratch30k-20260916.tar
```

如有私有 Registry，也可以用 `docker tag`、`docker push`，真机用 `docker pull`；
此时无需传 tar。

## 8. 开发机：传输镜像、checkpoint 和缓存

以下 `<robot-user>`、`<robot-ip>` 和目标路径需要替换：

```bash
rsync -av --info=progress2 \
  /home/descfly/groot-transfer/pi05-groot-server-scratch30k-20260916.tar* \
  <robot-user>@<robot-ip>:/data/groot-transfer/

rsync -aH --info=progress2 \
  /home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/newdata_v1_scratch30k_20260916/best_checkpoint/ \
  <robot-user>@<robot-ip>:/data/groot/checkpoint/

rsync -aH --info=progress2 \
  /home/descfly/workspace/Isaac-GR00T/outputs/dual_franka_wuji/models_http/ \
  <robot-user>@<robot-ip>:/data/groot/models_http/
```

`-aH` 用于保留 Hugging Face 缓存中的链接/硬链接结构。中断后可用同一命令续传。

## 9. 真机：准备代码和基础运行环境

先安装并验证 Docker、NVIDIA driver 和 NVIDIA Container Toolkit。至少确认：

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

克隆代码并切到部署分支：

```bash
git clone <PI05_GIT_URL> /home/<robot-user>/Pi05
cd /home/<robot-user>/Pi05
git checkout feat/fr3-wuji-groot-deployment
```

Pi05 控制侧使用项目 Python 3.11 `.venv`，不要叠加 Conda：

```bash
bash deploy/fr3_wuji/setup.sh
```

确认真机已有原控制镜像：

```bash
docker image inspect franka-upper-body-teleop:latest >/dev/null
```

如果没有，需按原 Pi05/FR3 部署流程构建，或从原真机环境单独 `docker save/load`。

## 10. 真机：校验并加载推理镜像

```bash
cd /data/groot-transfer
sha256sum --check pi05-groot-server-scratch30k-20260916.tar.sha256
docker load --input pi05-groot-server-scratch30k-20260916.tar
docker image inspect pi05-groot-server:scratch30k-20260916 >/dev/null
```

创建真机配置（该文件已由同目录 `.gitignore` 排除）：

```bash
cd /home/<robot-user>/Pi05
cp deploy/fr3_wuji_groot/.env.example deploy/fr3_wuji_groot/.env.groot
```

将 `deploy/fr3_wuji_groot/.env.groot` 改为：

```dotenv
GROOT_IMAGE=pi05-groot-server:scratch30k-20260916
GROOT_CHECKPOINT=/data/groot/checkpoint
GROOT_MODEL_CACHE=/data/groot/models_http
GROOT_PORT=8766
```

重复第 5、6 节的 `compose config`、容器 `--check-only`、启动、
`--server-check` 和假观测推理。目标机重新算出的模型指纹必须仍为：

```text
cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c
```

## 11. 真机：Pi05 无硬件预览

模型测试通过后执行：

```bash
bash deploy/fr3_wuji_groot/run.sh --check \
  --output /home/<robot-user>/Pi05/logs/weight_motion_eval/groot-migration-check
```

该步骤只生成 runtime 和启动命令预览，不连接设备。原 Pi05 路径也应单独检查：

```bash
bash deploy/fr3_wuji_fast/run.sh --check
```

## 12. 现场人工确认后才允许真机试运行

以下不是远程迁移的自动步骤。必须在现场具备物理急停、清空工作空间、确认初始姿态，
并由用户明确决定后，才进行单个 16 步动作块：

```bash
bash deploy/fr3_wuji_groot/run.sh --execute --supervised-trial \
  --server http://127.0.0.1:8766 \
  --expected-model-sha256 cf9fd5ab49b8dab8f27bccd02bfe280f8cd0914247c3a5551754adcefd66889c \
  --rounds 1 \
  --start-cameras \
  --finish-policy disable
```

首轮禁止使用多轮、后台无人值守或 `finish-policy hold`。

## 故障定位顺序

1. `docker compose config` 失败：检查 `deploy/fr3_wuji_groot/.env.groot` 和绝对路径。
2. `--check-only` 失败：检查 checkpoint 是否传完整、缓存 revision/链接是否完整。
3. 容器 unhealthy：查看 `docker compose logs groot-policy`，重点检查 GPU、显存和缓存。
4. `--server-check` 失败：检查端口、`warmed_up` 和模型 SHA256。
5. 假观测失败：保持硬件关闭，先解决模型输出或超时。
6. Pi05 `--check` 失败：处理真机 `.venv`、控制镜像和 overlay，不要尝试 `--execute`。
