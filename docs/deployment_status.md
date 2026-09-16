# 部署目录整理与本机检查（2026-09-16）

## 最新：两个执行策略与两份 π0.5 权重独立选择

- 新权重位置为 `checkpoints/pi05/30000`。state/actions 统计均为 64 维，末 10 维的 mean/std/q01/q99 全部为零。
- 使用现有 64→54 适配实际加载 30000 并完成假观测推理，输出有限 `(50, 54)`。记录在 `logs/pi05-model-adapters-20260916/30000-check.json` 和 `30000-smoke.log`。没有连接机器人。
- 对外执行策略为 `slow`、`normal`，两者都可通过 `--model 19999|30000` 选择权重；服务使用 `bash deploy.sh serve 19999|30000`，不再绑定执行策略。
- 模型维度、补零、关节重排和动作表示转换留在适配层。慢速/正常速度接收相同的 54 维绝对动作接口。
- 新入口默认 standard 准入，换模型不会自动放大规划阈值；慢速显式 `--planning-limits extended` 可选择旧 30000 的 1.5/0.2 rad 参数。旧 `slow64` / `30000` 别名保留原 extended 参数。
- 本轮相关回归 **86 项全部通过**，覆盖四种模型/策略组合、30000 适配、RTG 与正常速度执行回归。此前 4 项端点阈值断言差异保留，没有通过修改设备控制参数消除它们。
- 当前运行依赖检查仅缺实际相机配置和专用控制器构建，两份权重均已就绪。64 维权重搭配正常速度已做离线接口/启动配置检查，未做真机验证。

## 前次记录：并列预设与 19999 权重验证

- 当时对外预设为 `slow54`、`slow64`、`normal`，配置集中在 `deployment/profiles.py`；不再以训练步数命名方案。
- 慢速和正常速度是两个独立执行器，均依赖 `deployment/hardware.py`，正常速度不再引用慢速执行入口获取设备逻辑。
- `deployment/models/joint54.py` 和 `joint64.py` 分别负责模型接口，统一服务在 `deployment/server.py`。
- 64 维慢速规划采用显式会话配置，已删除原 `planning_profile.install()` 的全局函数替换；54/64 维配置可以在同一进程中分别构造，不会互相污染。
- 用户复制的 `checkpoints/pi05/19999` 已设为 `slow54` 和 `normal` 的共同默认权重。自带统计为 54 维，实际资产路径为 `assets/fr3_wuji/tomato_42_102_125/norm_stats.json`。
- 已执行 `bash deploy.sh serve slow54 --smoke-only`，实际加载权重并完成假观测推理，输出 `(50, 54)` 且全部有限；进程已退出，没有启动 WebSocket 常驻服务或连接机器人。
- 新增 9 项架构测试通过，覆盖三方案选择、共享设备层、54/64 维拒绝混接、规划配置隔离及检查模式不加载模型。
- 本轮最终回归 **344 项通过、4 项失败**；失败仍为原有到位阈值断言差异，设备限值没有变更。新结构的 172 个 Python 文件和 11 个 Shell 入口语法检查通过。

本次权重检查记录：`logs/deployment-architecture-20260916/19999-check.json` 与 `19999-smoke.log`。

以下为首次目录清理记录，已更新当前缺失项。

## 已完成

- 删除训练入口、训练配置集合、数据加载器、优化器、训练 checkpoint 写入以及数据转换/合并工具。
- 删除无关 ALOHA/LIBERO/DROID 示例和仿真子模块；部署安装不再包含 LeRobot、W&B、数据集和仿真依赖。
- 将 FR3/Wuji 的推理配置与 checkpoint 归一化读取独立到 `src/openpi/inference/`，模型共享分片工具移到 `src/openpi/shared/sharding.py`。
- 保持 19999 的 54 维 arm-delta/hand-absolute 契约，保持 30000 的 64→54 维绝对动作适配。
- 修正扁平目录下的模块导入、启动脚本、ROOT 和默认权重路径，新增根目录 `deploy.sh`。
- 遥操依赖默认改为 `/home/descfly/llx/gello_upper_body_teleop`，允许通过环境变量覆盖。未修改该仓库。
- 修正复制的虚拟环境引用原 openpi 源码的问题，并按新的 `uv.lock` 离线同步部署依赖。
- 旧训练日志、统计和元数据原样保存在 `archive/training/`；已有运行/诊断日志保留。

## 检查通过

- 本机已有 `franka-upper-body-teleop:latest` 控制镜像。
- 本机已有 Python 3.10 遥操环境和 `wuji_sdk` / `wuji_sdk.libs`。
- 遥操仓库的双臂与双手 54 维关节限位能正常读取。
- 在临时目录应用控制器补丁成功，当前遥操仓库源码与补丁匹配。临时目录已清理；这不是已完成编译。
- 当前部署环境识别 `cuda:0`，GPU 矩阵乘法、BF16 卷积和真实 tokenizer 检查通过。tokenizer 已从本机缓存复制到部署缓存，无需为本次检查下载。
- 三套执行器和两个模型服务的 `--help` 均能从统一入口启动。
- 157 个 Python 文件语法检查、11 个 Shell 入口语法检查通过；`uv lock --check --offline`、`uv pip check` 和变更空白检查通过。
- 本次离线回归 **335 项通过，4 项失败**。新增部署隔离、路径和模型动作契约检查及 30000 适配检查全部通过。

软件环境报告：`logs/deployment-audit-20260916/environment.json`。

## 仍需准备的运行材料

| 项目 | 状态 / 操作 |
| --- | --- |
| 19999 权重 | 已就绪：`checkpoints/pi05/19999`，真实加载及假观测推理通过 |
| 30000 权重 | 已就绪：`checkpoints/pi05/30000`，真实加载及假观测推理通过 |
| 相机配置 | 遥操仓库只有 `cameras.yaml.example`，缺实际 `data_collection/config/cameras.yaml`；可填写本目录 `config/cameras.example.yaml` 的副本，并通过 `OPENPI_CAMERA_CONFIG` 指向它 |
| 专用控制器 | 缺 `.deployment/oneshot-overlay` 的本机构建；按 README 运行 `prepare-controller` 和 `build-controller` |

首次目录清理未加载权重；后续已验证 19999 和 30000 权重（见顶部更新）。两次工作均未连接机器人、启动控制器/相机或发送真机动作。

## 原有回归差异

4 项失败均来自整理前已有的断言与当前控制参数不一致：

- `oneshot/test_core.py` 的两项测试仍要求机械臂端点误差大于 0.03 rad 时不能到位。
- `oneshot/test_slider.py` 的两项测试仍要求手部起点 0.09 rad、终点 0.10001 rad 不能到位。
- 当前 `oneshot/limits.py` 的 `ARM_ENDPOINT_TOLERANCE_RAD`、`SLIDER_POSITION_TOLERANCE_RAD`、`HAND_ENDPOINT_TOLERANCE_RAD` 均为 **0.25 rad**。Git 原版本已是该值。

本次没有修改这些控制常量、核心控制逻辑或上述测试断言，也没有把失败标成跳过。后续需要明确应采用的到位容差后，再统一实现、测试和旧文档；不应把本次整理视为整套执行系统已完成验收。
