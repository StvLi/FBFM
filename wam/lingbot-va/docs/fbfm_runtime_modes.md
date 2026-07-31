# LingBot-VA 中 NONE / RTC / FBFM 的统一运行语义

> 本文用于说明运行语义并保留原始实验审计坐标；其中的绝对路径和历史
> commit 仅是 provenance，不是当前复现入口。环境构建与启动命令以
> [`../README.md`](../README.md) 和仓库根 `README.md` 为准。

本文对应代码基线 `3604b457a24485cddf326a997e48955b7ca6b548` 上的修复，目标是落实
Theory-to-Implementation Handover 中的两点：恢复 previous-action constraint；让执行期真实
state feedback 影响**当前正在运行的** flow-matching chunk，而不是只在推理开始时取一次快照。

## 1. 唯一框架，三种掩码

三种模式使用相同的 `VA_PrevChunkAdapter`、`ChunkConstraintContext`、flow scheduler、两路
WebSocket 和 Robotwin rollout。target tensor 在三种模式中也相同，模式控制 solver 可见的 mask；
只有 `FBFM` 创建和推进独立 feedback VAE，`NONE/RTC` 收到 feedback 消息后直接忽略：

| 模式 | state mask | action mask |
|---|---|---|
| `NONE` | 全 0 | 全 0 |
| `RTC` | 全 0 | RTC 的 hard + soft mask |
| `FBFM` | 已到达真实 latent slot 为 1 | 与 RTC 逐元素完全相同 |

首个 chunk 没有 previous action，因此 RTC/FBFM 的 action mask 自然退化为 0。`Feedback` 仅作为
旧配置的兼容别名解析为 `FBFM`，不再是一条独立实现路径。

## 2. RTC 时间坐标

令动作预测总长为 `H`，固定重叠边界为 `d` 个控制步，每次换块执行长度为 `s`：

- `[0,d)`：已经/必将由旧 chunk 执行，权重为 1；
- `[d,H-s)`：重叠但尚未执行，使用论文的 `LINEAR` 或 `EXP` 软衰减；
- `[H-s,H)`：新动作自由生成，权重为 0。

必须满足 `0 <= d <= H-s`。本数学方法实验不测量 wall-clock 延时，也不动态修改 `d`。当前
`H=32,d=16,s=16` 恰好没有 soft 区，测试还覆盖
`H=32,d=4,s=12` 的非退化情况，并逐元素对照仓库核心 `RTCProcessor`。

这里的一个 action 是完整的 14D 或 16D 向量，不是标量。`H/d/s` 只数时间步；模型内部 target
为 `(B,D,F,N,1)`（Robotwin 为 `D=30`，其中 16 个有效 channels）。RTC 先生成
`(B,1,F,N,1)` 时间权重，再与 LingBot 原生 channel mask 相乘得到 `(B,D,F,N,1)` 的有效 mask；
因此同一时间步的 16 个真实 action 分量使用同一个 RTC 权重，14 个未使用 channels 始终为 0。
上一 chunk 的 public/物理动作会先经过 LingBot 原生 `preprocess_action` 回到归一化的 30 维模型
坐标，未使用 channels 与原模型一样清零，然后才作为 action-flow target；不能把反归一化动作直接
与 solver sample 作差。

## 3. state feedback 的实时语义

Robotwin 每 4 个控制 observation 形成 1 个 LingBot-VA video latent slot。服务端为每个 chunk
维护线程安全、可关闭、单调版本号的 `ChunkConstraintContext`。反馈携带
`observation_action_step`，编码后通过

`global_slot_id = feedback_target_frame_st_id + local_slot`

落到当前 chunk。video flow solver 在**每个 denoising step 之间**：

1. 排空 rank-0 反馈队列并向所有 rank 广播同一批反馈；
2. 各 rank 在 solver 边界顺序执行 VAE 编码，避免并发 CUDA/cache/collective；
3. 更新 context 的 target、mask 和 version；
4. 重新 snapshot state constraint，再执行下一 solver step。

重复 slot、窗口外 slot 和已关闭 context 都会拒绝更新。action flow 同样从 context snapshot，但
执行期间迟到的反馈不会误入 action solver；它会保留到下一 video solver 边界。

每个 combined inference request 还携带 `feedback_window_start_action_step`。所有 feedback 无论模型
当时是否运行都只进入同一个 FIFO；`observation_action_step <= window_start` 的数据属于刚写入 KV
cache 的历史，只推进 feedback VAE 而不置未来 state mask。只有严格晚于 cutoff 的 observation
才能约束当前预测 chunk，从而避免把历史 latent 错配到未来 `global_slot_id`。

## 4. 确定性伪异步与 reset

客户端使用两个 WebSocket：主连接承载 inference，控制连接按顺序承载 feedback、solver grant
和 reset。后台线程只负责避免阻塞控制连接，不代表实验使用 wall-clock 并发语义。
`DistributedModelWrapper` 只允许推理线程发起分布式 collective；所有 feedback 都进入队列，
reset 先发取消标记并等待 solver-step 边界，随后在互斥区清理模型、VAE 和 KV cache。

Robotwin 不再跳过仿真动作，也不根据实际推理速度决定时序。每个 simulation step 执行后，
客户端先发送该步产生的 feedback，再按固定整数比例授权 video-flow solver evaluations，并等待
这些虚拟步确实完成。默认 16 个 simulation steps 恰好释放 26 个 video solver evaluations；
action flow 随后完成，再在确定性边界切换动作 chunk。

首个异步转换不再做无反馈 warm-up。初始 inference 后，客户端只把已存在的初始 latent 和 LingBot
conditioned action frame 写入真实 KV，然后立即启动下一个重叠 chunk；旧 chunk 的 suffix 在该 solver
运行期间执行，产生的 observation 作为动态反馈进入当前 chunk。每轮执行得到的 4 个 observation 和
对应的 1 个 action frame 只暂存一次，并在**下一轮**启动前写入真实 KV。这样 solver-start history
\(\mathcal H_t\) 与 chunk 内动态反馈 \(\mathcal F_{t,k}\) 严格分离，不会把同一真实 observation
重复写入 history，也不会让 video/action KV 的时间维错位。

默认两 latent-frame chunk 的前两次衔接如下：

| launch | 启动前写入真实 KV | active generation | 同期真实执行 | dynamic state feedback |
|---|---|---|---|---|
| chunk 1 | 初始 \(z_0\) + conditioned action frame 0 | global frames 1--2 | chunk 0 的 frame 1 suffix | \(z_1\to\) chunk 1 slot 0 |
| chunk 2 | \(z_1\) + 实际执行的 chunk 0 frame 1 | global frames 2--3 | chunk 1 的 frame 2 suffix | \(z_2\to\) chunk 2 slot 0 |

因此某个 \(z_i\) 先作为正在运行 chunk 的反馈出现，下一轮才成为真实历史；它不会在同一次 generation
中同时扮演 \(\mathcal H_t\) 和 \(\mathcal F_{t,k}\)。

## 5. 配置和消融

主要环境变量：

- `LINGBOT_VA_CONSTRAINT_MODE=NONE|RTC|FBFM`
- `LINGBOT_VA_RTC_DELAY`、`LINGBOT_VA_RTC_EXECUTION_HORIZON`
- `LINGBOT_VA_RTC_ATTENTION_SCHEDULE=LINEAR|EXP`
- `LINGBOT_VA_FEEDBACK_OBS_PER_STATE=4`
- `LINGBOT_VA_FEEDBACK_LIVE=0|1`
- `LINGBOT_VA_PSEUDO_VIDEO_SOLVER_STEPS=26`

`LINGBOT_VA_FEEDBACK_LIVE=0` 只用于 `FBFM-static` 诊断消融：反馈仍被接收、编码并持久化，但不
更新正在运行的 context。它不是第四种约束模式。

四组 Robotwin 命令：

```bash
script/run_constraint_ablation.sh NONE 1
script/run_constraint_ablation.sh RTC 1
script/run_constraint_ablation.sh FBFM-static 1
script/run_constraint_ablation.sh FBFM 1
```

RoboTwin 使用 NVIDIA Vulkan raster backend，并显式关闭 SAPIEN ray tracing；CuRobo planner
仍需要可见 CUDA。当前 RTX PRO 6000 部署让策略 server 和客户端共享 GPU 0；GPU 端到端测试前
必须先暂停 DreamZero x LIBERO。

## 6. 可审计验证

无需 checkpoint 的确定性 replay：

```bash
python script/replay_constraint_modes.py --H 32 --d 4 --s 12 --schedule EXP
```

它输出三个模式的完整 action mask、state 可见性和 context version，并内置断言：NONE 两种 mask
为零、RTC state 为零、RTC 与 FBFM action mask 完全相同、只有 FBFM 能看到实时 state slot。

CPU 回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_fbfm_bridge.py tests/test_async_transport.py
```

测试覆盖 NONE 数值退化、RTC 核心实现对齐、14D/16D 完整向量对齐、context 动态版本、slot
时序与 window cutoff、真实 history 单次提交和 video/action 时间维校验、静态消融、solver 边界
队列、双 WebSocket 并发和分布式 wrapper 队列路径。解析测试还逐项验证
`x1 = x - sigma*v`、endpoint VJP、论文的 `lambda_tau` 与负 `delta_sigma` 下的更新符号。
当前结果为 `39 passed`。

## 7. 论文关系

- RTC（*Real-Time Execution of Action Chunking Flow Policies*）定义 previous-action 的异步衔接、
  `H/d/s` 与 soft mask；这里直接复用其 action 约束。
- LingBot-VA（*Causal World Modeling for Robot Control*）提供因果 video-action、KV cache 与真实
  observation 重新落地的系统框架。
- DreamZero（*World Action Models are Zero-shot Policies*）同样说明 WAM 的视频未来与 inverse
  dynamics 关系，并展示真正实时化依赖系统级流水线。

因此，FBFM 不是替换 RTC：它等于“完整 RTC action constraint + chunk 内实时 state constraint”。
论文和理论交接材料位于历史 paper checkout（未随公开仓库分发）。
