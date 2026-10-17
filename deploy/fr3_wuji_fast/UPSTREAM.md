# 上游来源与本地适配

这是 FR3/Wuji 部署适配，不是未经修改的 Wuji 发行版，也不是新训练的模型。

## Wuji RTG 来源

- 仓库：<https://github.com/wuji-technology/wuji-openpi>
- 核对的本地 checkout commit：`be6b32540546fb4fdd274f5cc86a7f9a67148a4c`
- 参考文件：`packages/openpi-client/src/openpi_client/rtg_action_broker.py`
- 核对时源文件 SHA-256：`67df69f19acfb7644f1350ce73196237742aa3e032fb39fb813a984ee80e27df`
- 许可证：Apache-2.0；完整上游许可证保存在本目录 `UPSTREAM_LICENSE`。

`timeline.smooth_prefix` 的 cubic Hermite 计算适配自上游 `cubic_smooth_prefix`。本地重新实现
编排层，使既有 FR3/Wuji 设备进程继续拥有设备、期限检查和网关；运行时不依赖另一份
`wuji-openpi` checkout。

保留的 RTG 概念包括：50步动作块、30 Hz 消费、半块异步推理、延迟补偿和3节点 cubic
前缀引导；另提供 serial 模式。RTG 可以在旧块尚未播放完50步时替换动作块。

本地 RTG 的主要适配如下：

- 每个会话仅首次目标走既有受控接管。
- 独立进程把30 Hz节点线性插值为100 Hz设备帧。
- RTG offset 使用请求提交后的实际单调时钟耗时，向上取整到下一个模型节点；它不是推理频率设置。
- cubic guidance 从实际拼接 offset 开始，并锚定当前指令位置。上游 broker 先平滑前缀再跳过延迟步，
  可能把已平滑的部分整体跳掉；本地顺序有意不同。
- 19999 Fast的 `--arm-motion-limits` 默认 `disabled`，不再把首块或拼接后的机械臂目标投影到1.0 rad/s可达集，也不在正常回放中叠加
  Ruckig 1/3/30或网关1.2 rad/s变化率阈值；RTG、RTC、serial行为一致。显式传
  `--arm-motion-limits legacy` 才恢复这些旧的本地约束。原始预测继续写入运行记录。
- 首次接管仍从实测姿态走本地受控接近；关节位置、0.2 rad跟踪、接触力矩、20 ms时效、反馈健康、
  失联与停止检查保留。慢速和中速调用方不受19999 Fast默认值变化影响，30000v2 Fast包装默认legacy。
- 模型或观测缺失从 trigger 起最多等待1.5秒；预测还必须通过原有70 ms观测发送预算和1秒
  观测到接纳期限，并在实际拼接时再次核验。
- 推理过晚时只继续执行当前已接纳块的剩余部分，随后保持末姿态；不外推或重放陈旧动作。
- 设备桥使用单一 stream identity 和单调递增帧序号。每个原始预测都有独立 request ID 与哈希，
  100 Hz控制器在切换节点前核验。设备桥 plan hash 标识动作流，而不是预先计算的样条。
- 继续继承 slider 手部投影、限速、接触处理和设备参数；实际手部运动可能落后于目标。

## PI RTC 参考与本地实现

模型端 real-time chunking（RTC）的依据是 Physical Intelligence 的
[RTC 论文](https://arxiv.org/abs/2506.07339)、
[项目页](https://www.pi.website/research/real_time_chunking)与
[官方参考仓库](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)。本仓库只按这些
公开资料实现所需接口和推理期引导，没有复制大段上游实现，也没有增加对官方仓库的运行时依赖。

本地 RTC 必须显式选择 `--broker-mode rtc`，默认模式仍是 `rtg`。它由两部分组成：

1. 客户端把请求时仍有效的旧动作作为前缀，连同预测推理延迟发送给模型；模型在 JAX 推理中使用
   `soft_prefix_guidance`，按 schedule 和引导权重使新块前缀靠近旧动作。
2. 响应返回后，控制进程按请求发出至实际接纳的端到端单调时钟延迟向上取整，跳过已经过期的节点，再直接
   安装剩余动作。由于新块已经在模型端受过前缀引导，RTC 路径不再叠加 RTG cubic 拼接。

默认参数为 `--rtc-initial-delay-steps 5`、`--rtc-delay-buffer-size 10`、
`--rtc-prefix-attention-schedule exp`、`--rtc-max-guidance-weight 5`。协议中的相应字段是
`inference_delay_steps`、`prefix_attention_horizon`、`prefix_attention_schedule`、
`max_guidance_weight`；响应回执位于响应顶层 `rtc` 字段，请求和回执还会绑定源动作块 token、
编号和 SHA-256。过期前缀、回执不匹配或实测延迟越过已引导 horizon 都按 fail-stop 处理：整次执行
报错并进入既有设备停止路径，而不是仅拒绝本次切换后继续播放旧块。

RTC 当前只对19999的50×54动作、JAX推理路径开放。PyTorch路径、64维权重和旧服务不会宣告
RTC 能力；服务还须在监听前完成普通与 guided 推理预热。19999 Fast的运动限值选择独立于RTC：
默认不增加机械臂速度、加速度或jerk软件上限，`--arm-motion-limits legacy` 才恢复1/3/30和网关1.2。
手部限制以及关节位置、跟踪、接触、时效、反馈与停止检查继续生效，它们不是模型端RTC的替代物。
上述1/3/30和1.2是本地legacy配置，不是Wuji RTG或PI RTC协议承诺的硬件安全边界。

截至2026-10-17，当前验证状态以开发日志为准；目标工作站、GPU/JAX、ROS整链路和真机均尚未
完成验证，不能把默认取消动态软件上限解释为硬件无限速或安全验收。验证状态和建议命令见 [README](README.md) 与
[`docs/开发日志.md`](../../docs/开发日志.md)。
