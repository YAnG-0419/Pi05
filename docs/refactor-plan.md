# 部署目录重构计划（2026-09-16）

本文只记录**需要改什么、为什么、怎么验证**，不改变现有分层设计。
执行策略 × 权重的两轴设计是正确的，保持不变；本计划不触碰任何设备限值、
规划参数数值、控制频率或安全门禁逻辑。

状态：本文是待实施计划。本次仅修订文档；下文的 `robot_deploy` 命令、run YAML、
公共会话和迁移后的目录尚未实现。当前可用命令仍见根目录 README。

## 事实核对：硬件接口只有 54 维

先确认一个前提，后续命名建议基于此：

- 真机接口恒为 54 维：左臂 7 + 左手 20 + 右臂 7 + 右手 20。
- `experiments/weight_motion_eval/oneshot/core.py:26-27` 关节分组
  `ARM = [0:7, 27:34]`、`HAND = [7:27, 34:54]`；
  `core.py:47-50` 的 `vector()` 硬性要求 `(54,)` 有限值。
- 设备动作与反馈采用 54 维物理契约；IPC 中的包大小、序号位宽不代表关节维度。
- 30000 的 64 维处理封装在 `deployment/models/joint64.py`：输入重排、补零后归一化；
  输出反归一化后恢复物理关节顺序、丢弃末 10 维，出适配层即为 54 维。

**结论：`64` 是模型内部细节，不应出现在执行入口名里。**

---

## 问题 1：规划阈值预设复用了 profile 的名字（优先做）

### 现状

命令行的词汇是 `standard` / `extended`（`deployment/run.py:12`），
内部却用 `slow54` / `slow64`（`deployment/profiles.py:61`、
`experiments/weight_motion_eval/planner.py:68`）。
而 `slow64` 同时还是一个执行入口名（`profiles.py:43`）。同一个词指两件事。

### 后果（已核对代码，两条路径确实不等价）

| 入口 | engine | adapter | checkpoint | port | planning_profile |
|---|---|---|---|---|---|
| `deploy.sh slow64` | slow | joint64 | 30000 | 8002 | **slow64** |
| `deploy.sh slow --model 30000` | slow | joint64 | 30000 | 8002 | **slow54** |

除了入口名，engine、adapter、checkpoint、port 相同，planning_profile 不同。
差异落在 `planner.py:71-73`：

```python
if profile == "slow64":
    config["arm_raw_initial_delta_rad"] = 1.5          # 默认 0.3，5 倍
    config["max_smoothing_delta_rad"]["arm"] = 0.2     # 默认 0.03，6.7 倍
```

即：**同一份权重、同一个执行器，因为敲的入口名不同，首帧准入偏差相差 5 倍。**
这是 README 里写明的兼容性决定，不是 bug；问题在于名字没有表达出来。

### 改法

1. 规划阈值预设改名：`slow54` → `standard`，`slow64` → `extended`。
   涉及 `profiles.py`、`planner.read_config()`、`planner.validate_config()`
   （包括默认值与阈值分支）、`slow.py` 及相关测试。不能只改读取函数。
2. `PROFILES` 里 `slow64` 显式写成 `planning_profile="extended"`（`profiles.py:43`），
   让表格里那一列自解释。
3. `run.py:19` 的 `'slow64' if ... else 'slow54'` 映射随之删除，
   命令行值直接透传。

### 不改

- 1.5 / 0.2 与 0.3 / 0.03 这两组数值本身不动。
- `slow64` 作为**执行入口别名**继续保留，行为不变（仍为 extended）。
- `compose_profile` 中 `normal` 拒绝 extended 的约束保留（`profiles.py:63-64`）。

### 验证

```bash
bash fr3_wuji/run.sh -m pytest tests/test_parallel_profiles.py -q
```
`tests/test_parallel_profiles.py:129-138` 已覆盖四种组合的 planning_profile 取值，
改名后同步更新断言字符串即可；断言的**语义**不应改变。

### 风险

低。纯字符串改名，无行为变化，有测试兜底。

---

## 问题 2：设备会话在两条链路各写了一遍，且已经漂移

### 现状

`deployment/slow.py` 与 `deployment/normal/deploy.py` 重复实现了抢占锁、
模型 metadata 与 warmed_up 校验、起容器、等 `devices.sock`、取 inventory、
轮询 feedback、read-only 分支及清理。重复流程确实可以抽取，
但相同行数不能作为职责划分依据：会话参数和清理顺序存在策略差异。

### 已发生的漂移（这是要修的实际缺陷）

| | `slow.py:262-269` | `normal/deploy.py:281-287` |
|---|---|---|
| `devices.stop()` | try/except 包住 | 裸调用 |
| `stop-request.json` | 写，含 `physical_stop_confirmed` | **不写** |
| 异常入口 | `except KeyboardInterrupt` | `except BaseException` + report |

正常速度缺少的是**上层停止请求记录**，不是完全没有物理停止证据。
两条链路共用的 `oneshot/bridge.py:262-273` 在设备退出时写 `device-exit.json`，
包含 `physical_stop_confirmed`、`stop_errors` 等字段。

`devices.stop()` 返回的是停止请求的处理结果，不能等同于最终物理停止确认；
底层 `DeviceSession.close()` 还会执行反馈确认。请求失败或断连时必须保留“不确定”，
不能因为 RPC 返回、连接关闭或写出了 JSON 就标记确认成功。

异常处理也有实际风险：normal 的 `consumer.close()` 若抛错，后续显式 `devices.stop()`
会被跳过；停止调用本身抛错还可能遮蔽原始异常。底层仍有断连/退出清理，
不能据此断言机器人一定不停，但上层清理应保证各阶段独立尝试并记录失败。

### 改法

新增 `deployment/session.py`，提供公共会话生命周期；建议命名 `DeploymentSession`，
避免与底层已有的 `DeviceSession` 混淆。不包含轨迹生成和执行调度：

```
DeploymentSession(session_config, model_spec, output)
  → preflight_owners
  → flock 独占锁
  → ws 连接 + metadata 校验 + warmed_up
  → tempdir ipc + Children 启动 + 等 devices.sock（45s）
  → RemoteDevices + inventory + feedback 首帧
  → 退出时：独立尝试停止、记录请求结果、关闭连接和子进程
  → 保留底层 device-exit.json 的最终确认语义及失败记录
  yield (devices, ws, model, limits, names)
```

- `slow.py` 与 `normal/deploy.py` 各自只保留 consumer 构造、
  `observe.main(...)` 调用、hold 循环与自己的收尾文案。
- 预览使用独立的纯配置/命令构造函数，不进入设备会话；只读模式使用相同的
  会话生命周期，但不连接模型、不使能、不下发。消除为预览伪造 argparse 字段的写法。
- 用 dataclass 显式传递会话配置。保留 slow 的会话速度 0.7、normal 的 1.0 和
  normal 网关上限 1.2 rad/s；保留各自 finish_policy 默认值、准入方式与等待间隔。
  不得抽取后统一成某一条链路的默认值。
- 明确退出顺序：先停止控制 worker/运动，再处理可能阻塞的记录器与容器清理；
  consumer 清理异常不能阻止后续停止尝试，清理异常不能丢失原始故障。
- 两种策略都记录 `stop-request.json`，并保留底层退出报告；日志统一与异常路径可靠性
  是本条收益，不改变物理停止确认条件。

### 不改

- 两个执行器保持独立。timeline / consumer / planner 用法本就不同，
  不做任何合并。
- 45 秒超时、轮询间隔、finish_policy 语义不变。

### 影响面

`deployment/slow.py`、`deployment/normal/deploy.py`、
新增 `deployment/session.py`，以及配置/命令构造辅助函数和生命周期测试。
不以减少到某个行数作为验收标准。

### 验证

```bash
bash fr3_wuji/run.sh -m pytest tests/ fr3_wuji_fast/ fr3_wuji_30000/ experiments/ -q
bash deploy.sh slow   --check
bash deploy.sh normal --check
bash deploy.sh slow   --model 30000 --check
bash deploy.sh normal --model 30000 --check
```
现有 `--check` 不连设备、不加载模型，但会写新日志目录，且依赖本地控制器 overlay、
SDK 路径与外部仓库配置。当前缺 `.deployment/oneshot-overlay/source-manifest.json`，
不能直接运行上述命令取得完整预览；先用测试夹具取得基线，实际环境就绪后再核对。

预览包含随机容器名和输出路径，应固定测试输入或归一化这些字段后比较命令结构、
挂载、速度、准入与控制参数，不能逐字节比较两次 CLI 运行结果。

增加无硬件的生命周期测试：锁占用、metadata 不匹配、启动失败、反馈超时、
KeyboardInterrupt、consumer 清理抛错、stop 超时/断连、正常结束及 hold/disable。
验证清理顺序、原始错误留存、不虚报物理停止成功，以及 check 不启动任何设备会话。
已有测试基线包含 4 项端点阈值断言差异（见 `docs/deployment_status.md`）；
实施前复核基线，要求新测试通过且无新增失败，不通过改设备限值消除旧失败。

### 风险

中。改动落在安全握手与退出路径上。先完成离线异常测试和预览比较，
再进行现场只读反馈验证；只读验证需要设备，不能称为“不接真机”的检查。
只读通过也不等于运动行为已完成验收。

---

## 问题 3：目录名与真实架构不符

### 现状

生产设备层、执行策略和诊断逻辑混在 `experiments/weight_motion_eval/oneshot/`。
多个 `deployment/` 模块依赖它，目录名称与实际职责不一致：

```
deployment/hardware.py    → experiments...oneshot.qualification, reference
deployment/slow.py        → experiments...oneshot.{ipc,limits,qualification}
deployment/normal/*.py    → experiments...oneshot.{core,ipc,live,limits,continuous}
```

物理安全限值 `ARM_TRACKING_TOLERANCE_RAD`、`HAND_KP/KD`、`SLIDER_CURRENT_A`
埋在 `experiments/weight_motion_eval/oneshot/limits.py`。

模型与策略独立的分层保留；迁移时需进一步区分设备生命周期、执行策略和运行诊断，
不能把 oneshot 整体换名为 devices。

### 改法：先列依赖清单，再按职责迁移

迁移清单必须覆盖直接 import、间接 import、动态加载、子进程入口及配套资源。
建议归属如下，实际移动前逐文件确认依赖闭合：

| 当前职责/模块 | 目标归属 |
|---|---|
| IPC、bridge、devices、hand、ROS 边界及它们的设备依赖 | `deployment/devices/` |
| `live.py`、`continuous.py`、`runner.py` 中慢速执行逻辑 | 慢速策略包；迁移时统一处理当前 `slow.py` 入口 |
| `core.py` 中两种策略共用的动作契约、守卫；共用 StopFlag 等 | 公共运行模块，不能让 normal 继续依赖慢速策略实现 |
| `hand_trace.py`、`poll_timing.py`、`loop_diagnostics.py` | 生产诊断模块，随调用方迁移 |
| `planner.py` 及其部署参数 | `deployment/planner.py` 及明确的配置位置 |
| `reference.py` 中部署所需外部依赖访问 | `deployment/reference.py` |
| `oneshot/limits.py` | `deployment/limits.py`，数值不变 |
| `cli.py` 的 `safe_output()` | 提取为公共路径辅助函数；实验 CLI 本体留在原处 |
| 控制器构建脚本、prepare_overlay、record_build、deadline.hpp | 一起归入构建工具目录，保留资源相对关系 |

以下独立分析/试验入口可留在 `experiments/`，更新为调用生产模块：
`evaluate_checkpoint.py`、`rehearsal.py`、`preview.py`、`capture.py`、
`oneshot/analyze_hand_trace.py`、`check_*.py`、`probe_hands.py`、
`read_hand_status.py`。

特别注意：`hand.py` 直接 import `hand_trace.py` 和 `poll_timing.py`，
`bridge.py` 使用 `loop_diagnostics.py` 并以模块路径启动其子进程；它们不是仅用于离线分析。

测试按被测职责归档，同时迁移 conftest 和跨测试引用。
`fr3_wuji_fast/`、`fr3_wuji_30000/` 的转发壳先保留为兼容接口；
只有确认脚本、测试及对外入口不再依赖后，才另行决定删除。
不能仅凭文件只有数行就判定可删。旧命令别名在本轮保持行为不变。

### 必须注意：不是纯 git mv

模块路径被**硬编码进 docker 启动命令和入口脚本**，是字符串，
import 检查拦不住，错了要到起容器时才暴露：

- `deployment/hardware.py:154` `python3 -m experiments...oneshot.ros_boundary`
- `deployment/hardware.py:159` `experiments.weight_motion_eval.oneshot.bridge`
- `deploy.sh:26` `-m experiments.weight_motion_eval.oneshot.prepare_overlay`
- `deploy.sh:27` `experiments/weight_motion_eval/oneshot/build_overlay.sh`
- `bridge.py` 中启动 `loop_diagnostics` 的模块路径，以及 build_overlay 中的 `record_build`
- `hardware.preflight_owners()` 中的 `oneshot.bridge` 进程/容器匹配规则：
  必须同时识别旧、新入口，避免迁移后漏检已有设备拥有者
- `reference.py`、`cli.py`、`prepare_overlay.py` 等的 `__file__.parents[...]`，
  Shell 相对根目录计算，以及 `deadline.hpp` 资源查找

另外 `deployment/slow.py:205` 与 `normal/deploy.py` 中
`ROOT / "experiments/weight_motion_eval/config.yaml"` 是路径字符串，
`config.yaml` 随 planner 一起迁移时需同步。

### 顺带处理的小项

- `deployment/server.py:23` 的 `compose_profile('slow', args.model)`：
  服务端不需要执行策略，硬塞 `'slow'` 只为取得 Profile。
  改为直接用 `MODELS[args.model]`（ModelSpec 已有 adapter/checkpoint/port）。
- `deployment/hardware.py:33` `write_runtime(args, ...)` 直接吃 argparse namespace，
  设备层因此耦合 CLI 字段集。随问题 2 一起改为接收 dataclass。
- `deployment/models/joint64.py:8` 的 `PROMPT` 不在正式服务路径使用
  （server 用的是 `fr3_wuji.observation.PROMPT`），但测试直接导入 joint64 后引用它。
  先将测试改用公共任务文本，再移除重复常量，不能随转发壳直接删除。
- `joint54.checkpoint_contract` 与 `joint64.check_checkpoint` 的 norm_stats
  校验逻辑逐行重复，只差维度与补零检查，可合成一个带维度参数的校验器。

### 验证

```bash
bash fr3_wuji/run.sh -m pytest -q                 # 全量
bash deploy.sh check                               # 本地文件/镜像/包/限位
bash deploy.sh slow --check && bash deploy.sh normal --check
# 按问题 2 的前置条件和归一化规则比较预览；迁移路径属于需审核的预期差异
```
`tests/test_deployment_layout.py:55-60` 已覆盖「任意 cwd 下能定位项目」，
迁移后应继续通过。

另需覆盖：旧/新入口独占识别、构建资源定位、子进程模块可发现、兼容命令可用。
适用时在无设备、无网络的容器中验证模块导入和资源存在，不调用设备初始化。
明确记录允许改变的模块路径，其余控制参数与挂载权限不得悄然变化。
全量测试按问题 2 的已知失败基线核对；预览不证明容器内部模块能够运行。

### 风险

中。涉及间接依赖、进程独占识别、构建资源和退出路径。
仅 import 检查或命令预览不足，需结合异常测试、容器内静态验证和现场只读反馈验证。

---

## 执行顺序

| 步骤 | 内容 | 风险 | 前置 |
|---|---|---|---|
| 1 | 规划阈值改名 `standard` / `extended` | 低 | — |
| 2 | 抽公共会话，统一停止请求记录和异常清理 | 中 | 先留离线基线、补生命周期测试 |
| 3 | 按职责迁移生产依赖，保留兼容壳 | 中 | 步骤 2 完成，逐模块迁移清单已核对 |
| 4 | 实现 run 配置和 `robot_deploy` 三个入口 | 中 | 本文配置契约明确，复用前述会话与策略 |

每步独立可提交、可回退。步骤 2、3、4 完成后先通过离线回归，再在现场执行
只读反馈验证（不使能、不下发）；正式执行的准入要求不变。

## 统一入口：动作由命令选择，策略由 run 配置选择

以下是待实现接口，名称按用户提出的形式：

```bash
robot_deploy check   --run runs/pi05_19999_tomato_slow.yaml
robot_deploy serve   --run runs/pi05_19999_tomato_slow.yaml
robot_deploy execute --run runs/pi05_19999_tomato_slow.yaml
```

不再新增 `robot_deploy slow` / `normal` 子命令。slow、normal 是执行策略，
check、serve、execute 是操作。同一模型的服务不因策略不同而产生一套新实现。

| 命令 | 职责 | 边界 |
|---|---|---|
| `check` | 解析 run 与 spec，检查配置版本、输入/输出契约、适配器、权重文件和归一化统计、策略兼容性 | 不连接机器人或模型服务，不启动 Docker、ROS、SDK、相机，不加载 GPU 模型 |
| `serve` | 读取模型 spec、权重和服务地址，加载模型、假观测预热、启动推理服务 | 不获取硬件独占锁，不启动设备；预热失败则不提供就绪服务 |
| `execute` | 验证配置、核对服务模型身份、检查本机硬件运行前置条件、获取独占并运行所选策略 | 需要已启动的模型服务；不自动切换模型、启动另一服务或降低准入要求 |

新 `check` 与当前 `deploy.sh slow --check` 并不完全等价：后者还需要 overlay 才能生成
硬件命令预览。新 `check` 以可移植的离线契约校验为边界，缺本机硬件安装不妨碍模型/spec
检查，但必须明确输出“硬件运行就绪未验证”，不能把离线通过报告成真机就绪。
本机镜像、SDK、overlay 检查及命令预览保留为单独诊断能力，并在 execute 前强制执行。
已提供的 spec 文件缺失、模型资产不完整或契约不匹配仍应使 check 失败。

### 三类配置独立

```text
runs/       本次运行：选择设备、模型、任务文本、服务地址、执行策略及准入方式
models/     模型 spec：模型结构、权重位置、适配器、归一化、输入需求及输出语义
robots/     设备 spec：设备地址、关节名称/顺序、相机角色与标定/配置引用、底层运行依赖
```

这些目录和 YAML 尚未创建。模型 spec 内部再引用权重自己的 norm_stats，
不把另一份模型的统计复制成公共配置。设备 spec 不包含训练补零维度。
run 引用路径相对于 run 文件解析，spec 内部路径相对于 spec 文件解析，不能依赖启动 cwd。

示意 `runs/pi05_19999_tomato_slow.yaml`：

```yaml
schema_version: 1
robot: ../robots/fr3_wuji_workcell_01.yaml
model: ../models/pi05_19999.yaml
task:
  prompt: >-
    Pick up a tomato truss with the right hand, then pick a cherry tomato
    with the left hand and place it in the left basket.
service:
  bind_host: 127.0.0.1
  port: 8001
  client_uri: ws://127.0.0.1:8001
execution:
  strategy: slow
  planning_limits: standard
  finish_policy: disable
  admission:
    mode: supervised_trial
  options:
    continuous: true
    rounds: 50
    replan_steps: 20
    minimum_time_scale: 2.5
```

示意 `runs/pi05_19999_tomato_normal.yaml` 的设备、模型、任务与服务部分保持相同，
execution 部分替换为：

```yaml
execution:
  strategy: normal
  planning_limits: standard
  finish_policy: disable
  admission:
    mode: supervised_trial
  options:
    broker_mode: rtg
    rounds: 50
```

这两个示例显式选择 disable 和现有 supervised_trial 准入，不更改旧入口的默认值；
执行这类配置仍表示现场值守试运行。正式 qualification 模式需显式引用资格文件，
缺失时失败，不能自动降级为 supervised_trial。两个模式均保留现有验证逻辑。

正常速度启动示例：

```bash
robot_deploy check --run runs/pi05_19999_tomato_normal.yaml
# 终端一：保持模型服务运行
robot_deploy serve --run runs/pi05_19999_tomato_normal.yaml
# 终端二：条件满足时执行真机动作
robot_deploy execute --run runs/pi05_19999_tomato_normal.yaml
```

使用 30000 时，选择对应模型 spec；两模型同时提供服务时配置不同端口，
当前约定 19999 为 8001、30000 为 8002。执行策略和 planning_limits 不因换权重自动变化。
同一模型服务可供两种策略先后使用，设备独占仍禁止并发执行；更换策略先结束原会话。
`runs/dp3_tomato.yaml` 可作为未来同形式的配置名，但当前没有 DP3 适配器或资产，
不能仅凭创建此文件就声称支持 DP3。

### 配置与通信校验要求

- run 解析为类型明确的配置对象；不把任意 YAML 字段直接拼到命令中。
  校验未知字段、策略专属选项、端口、地址和路径，拒绝 normal + extended 等无效组合。
- 显式携带任务文本到观测请求，覆盖当前观测代码里的默认任务，不能只配置服务默认 prompt。
- 模型 spec 声明相机角色、颜色/数据类型、状态顺序/单位、动作表示、horizon 和时间间隔。
  当前两模型经适配后保持 `(50, 54)` 绝对关节位置契约，执行时间语义保持现状。
- check 不推理，因此不能证明权重可成功加载或模型行为正确；完整文件/契约检查与
  serve 的实际加载预热分开报告。spec 中若声明资产哈希，必须验证；
  manifest/norm_stats 哈希不等同于全部权重数据的完整性证明。
- execute 继续核对服务端 checkpoint/normalization 身份与 warmed_up；模型 spec 的契约
  需与实际适配器输出一致，不能只相信配置中的维度数字。
- 覆盖两模型 × 两策略、任意 cwd、无硬件依赖的 check/serve 路径、服务身份不匹配、
  无效配置和旧命令兼容性测试。check 测试应明确断言未创建设备/网络连接和硬件子进程。

## 明确不做

- 不合并两个执行器。
- 不改任何设备限值、规划参数数值、控制频率。
- 不改 `(50, 54)` 模型-执行契约。
- 不改 `src/openpi/`、`packages/openpi-client/`。
- 不动 `checkpoints/`、`archive/`、既有日志。
