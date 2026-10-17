# FR3 + Wuji 部署结构与参数

## 目录职责

| 目录 | 责任 |
| --- | --- |
| `fr3_wuji_runtime/` | 三种速度共用的控制器 overlay 核验、工作站清单、Docker 启动命令、设备占用检查、子进程清理、设备桥、ROS 网关和有期限的 Unix socket IPC。|
| `fr3_wuji_slow/` | 慢速规划、播放器、连续执行、现场入口及慢速边界。旧名 `oneshot` 只保留兼容入口和暂未迁出的设备/观测模块。|
| `fr3_wuji_medium/` | 中速规划与连续执行；`runtime/` 保存中速特有的速度检查、设备桥和 ROS 网关。|
| `fr3_wuji_fast/` | 30 Hz 节点播放和RTG/RTC/serial调度；19999默认直接回放，legacy模式提供旧Ruckig臂部平滑。|
| `fr3_wuji_models/model_*/` | 权重服务、输入输出适配和 checkpoint 契约；与速度方案分开。`registry.py` 集中模型默认端口、路径和采样频率。|

`deploy/fr3_wuji_30000*`、`deploy/fr3_wuji_25000*` 中的 `serve.py`/`contract.py` 与 `experiments/weight_motion_eval/oneshot` 中已迁移的模块仍是兼容导入。旧目录的执行包装与 shell 脚本保留。不要在兼容文件里新增实现；按上表修改新目录。旧名还出现在锁文件、overlay、ROS 节点名称和历史日志里，改这些名字会影响现有部署，故暂时保留。

## 运行入口

| 方案 | 命令 | 默认模型 | 核心参数 |
| --- | --- | --- | --- |
| 慢 | `bash deploy/fr3_wuji_slow/run.sh --check` | 19999，`:8001` | `--minimum-time-scale 2.5`、`--continuous`、`--rounds 50`、`--replan-steps 20`、结束 `hold` |
| 中 | `bash deploy/fr3_wuji_medium/run.sh --check` | `64-lora-30hz`（原 30000），`:8002` | `--model`、`--minimum-time-scale 0.625`、`--slow-after-seconds 25`、结束 `hold` |
| 快 | `bash deploy/fr3_wuji_fast/run.sh --check` | 19999，`:8001` | `--broker-mode rtg/rtc/serial`（默认rtg）、`--arm-motion-limits disabled/legacy`（19999默认disabled）、`--rounds 50`、结束 `disable` |

三个入口都可设 `--left-arm-ip`、`--right-arm-ip`、`--wuji-left-address`、`--wuji-right-address`、`--checkpoint`、`--uri`、`--output`、`--start-cameras`、`--finish-policy`。默认设备地址分别是 `172.16.0.2`、`172.16.1.2`、`192.168.1.110:7447`、`192.168.2.111:7447`。`--check` 仅生成配置；`--read-only` 读取设备反馈；真机下发必须显式 `--execute` 并使用 `--qualification` 或 `--supervised-trial`。慢、中另有 `--hand-control slider/strict`；快固定 slider。三种方案共用 `.deployment/oneshot-owner.lock`，不能同时占用真机。

| 模型 | 默认端口 | 默认 checkpoint | 时间轴 | 新适配目录 |
| --- | ---: | --- | ---: | --- |
| 19999 | 8001 | `checkpoints/19999`（慢/中）；快默认 `checkpoints/19999_269/19999` | 30 Hz | `model_19999/` |
| 30000 | 8002 | `checkpoints/30000` | 30 Hz | `model_30000/` |
| 30000v2 | 8003 | `checkpoints/30000v2` | 30 Hz | `model_30000v2/` |
| 25000 | 8004 | `checkpoints/25000` | 15 Hz | `model_25000/` |
| 25000-single | 8005 | `checkpoints/25000-single` | 15 Hz | `model_25000_single/` |

上述 checkpoint 是代码默认或相对路径；以现场真实文件为准，模型服务和执行端必须使用相同权重。
中速用 `--model 54` 选择表中的 19999；64 维分别使用 `64-lora-30hz`（30000）、
`64-full-30hz`（30000v2）、`64-full-15hz-ab`（25000）、`64-full-15hz-a`（25000-single）。
旧参数仍兼容；`64` 仅作为 `64-lora-30hz` 的别名。另支持 `--model 20hz`（20 Hz、默认端口 8006）。
慢速原入口和快速主入口仍使用原生 54 维适配。
其他权重与速度的组合以各模型包装脚本和契约为准。

快速入口的 RTC 必须显式指定 `--broker-mode rtc`，目前仅支持19999、50×54动作和JAX推理。
默认 RTC 参数为 `--rtc-initial-delay-steps 5`、`--rtc-delay-buffer-size 10`、
`--rtc-prefix-attention-schedule exp`、`--rtc-max-guidance-weight 5`。模型端对旧动作前缀做
soft prefix guidance，执行端按实际延迟跳过过期节点；RTC 路径不再叠加 RTG 的 cubic 拼接，
且19999 Fast默认不经过1 rad/s速度投影或Ruckig。RTG、RTC、serial共用同一运动限值模式；显式传
`--arm-motion-limits legacy` 才恢复1 rad/s、3 rad/s²、30 rad/s³及网关1.2 rad/s。
关节位置、0.2 rad跟踪、接触、20 ms时效、反馈健康/失联和停止检查仍生效。服务响应回执位于顶层 `rtc` 字段；运行期的
过期前缀、回执不匹配或实测延迟超过已引导 horizon 会终止整次执行并走设备停止路径，而非只跳过
本次换块。64维快速入口只复用执行层，不因此获得 RTC 支持。

## 轨迹与边界参数

| 参数 | 慢速 | 中速 | 快速 |
| --- | ---: | ---: | ---: |
| 模型节点 / 设备下发 | 30 / 100 Hz | 30 或 15 / 100 Hz | 30 / 100 Hz |
| 最低时间倍率 | 2.5 | 0.625 | 1.0 原时间轴 |
| 臂部正常回放动态软件限值 | 目标0.42 rad/s及规划约束 | 目标1.68 rad/s及规划约束 | 19999默认无速度/加速度/jerk上限；legacy为1/3/30 |
| 臂部跟踪误差 | 0.08 rad | 0.2 rad | 0.2 rad |
| 规划文件 | `slow/config.yaml` | `medium/config.yaml` | `fast/timeline.py`、`fast/limits.py` |

慢、中 `config.yaml` 还包含接近/播放的双臂双手速度、加速度、jerk，平滑次数、停稳时间与最大轨迹时长；其中 `hardware_output: false` 表示配置本身不能开启真机输出。中速连续模式会在累计运动约 25 秒后的下一段边界改用慢速轨迹参数；设 `--slow-after-seconds 0` 可禁用。19999 Fast只在 `--arm-motion-limits legacy` 下启用旧Ruckig 1/3/30与网关1.2；30000v2 Fast包装默认选择legacy。19999默认取消的仅是正常模型回放的机械臂动态软件限值，首次受控接近及其约束不变。公共 IPC 包限制64 KiB，运动帧仍由设备桥检查位置、跟踪、时效、反馈、接触和停止边界。该默认路径尚未完成目标工作站、ROS或真机验证。

## 相机与移交

头部相机由用户确认是 435，IP 为 `192.168.35.35`；两台腕部是 305。当前相机链路由参考遥操仓库按序列号启动并发布 ROS `cam0/1/2`，OpenPI 订阅图像；本仓库没有这三台现场序列号。当前参考启动配置按三台 435Le 写成，腕部 305 的启动驱动仍需按实际子型号核对。详细接口、现场缺项和同事移交条件见`docs/部署_2026-09-29.md`。

只复制本仓库即可继续改离线代码；要运行部署还需参考遥操仓库、ROS/SDK 环境、控制器 overlay、相机配置和 checkpoint。公共运行层目前保留原工作站的绝对路径与 CPU 集，需要在另一台工作站部署前核对。
