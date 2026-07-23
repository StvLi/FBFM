# LingBot-VA 中 NONE / RTC / FBFM 的统一运行语义

本文对应代码基线 `3604b457a24485cddf326a997e48955b7ca6b548` 上的修复，目标是落实
Theory-to-Implementation Handover 中的两点：恢复 previous-action constraint；让执行期真实
state feedback 影响**当前正在运行的** flow-matching chunk，而不是只在推理开始时取一次快照。

## 1. 唯一框架，三种掩码

三种模式使用相同的 `VA_PrevChunkAdapter`、`ChunkConstraintContext`、flow scheduler、两路
WebSocket、反馈编码和 Robotwin rollout。target tensor 在三种模式中也相同，模式只控制 solver
可见的 mask：

| 模式 | state mask | action mask |
|---|---|---|
| `NONE` | 全 0 | 全 0 |
| `RTC` | 全 0 | RTC 的 hard + soft mask |
| `FBFM` | 已到达真实 latent slot 为 1 | 与 RTC 逐元素完全相同 |

首个 chunk 没有 previous action，因此 RTC/FBFM 的 action mask 自然退化为 0。`Feedback` 仅作为
旧配置的兼容别名解析为 `FBFM`，不再是一条独立实现路径。

## 2. RTC 时间坐标

令动作预测总长为 `H`，推理实际延迟为 `d` 个控制步，每次换块执行长度为 `s`：

- `[0,d)`：已经/必将由旧 chunk 执行，权重为 1；
- `[d,H-s)`：重叠但尚未执行，使用论文的 `LINEAR` 或 `EXP` 软衰减；
- `[H-s,H)`：新动作自由生成，权重为 0。

必须满足 `0 <= d <= H-s`。Robotwin 客户端在后台推理完成时记录实际执行步数，作为下一请求的
动态 `d`；配置值只作为首次/回退值。当前 `H=32,d=16,s=16` 恰好没有 soft 区，测试还覆盖
`H=32,d=4,s=12` 的非退化情况，并逐元素对照仓库核心 `RTCProcessor`。

这里的一个 action 是完整的 14D 或 16D 向量，不是标量。`H/d/s` 只数时间步；模型内部 target
为 `(B,D,F,N,1)`（Robotwin 为 `D=30`，其中 16 个有效 channels），mask 为
`(B,1,F,N,1)`，沿 `D` 广播。因此同一时间步的全部 action 分量总是使用同一个 RTC 权重。

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

## 4. 真异步传输与 reset

客户端使用两个 WebSocket：主连接承载后台 inference，控制连接承载 feedback/reset。服务端把同步
模型调用放入工作线程，事件循环可继续处理控制连接。`DistributedModelWrapper` 只允许推理线程
发起分布式 collective；所有 feedback 都进入队列，reset 先发取消标记并等待 solver-step 边界，
随后在互斥区清理模型、VAE 和 KV cache。

Robotwin 不再跳过若干循环来模拟异步，而是：后台生成下一 chunk；前台真实执行当前 chunk；每
4 步经控制连接发送 observation；边界接收新动作并换块。如果生成慢于 `s` 步，前台在安全边界
等待，不会并发调用同一 CUDA/cache 状态。

首个异步转换是因果 warm-up：初始 observation 已被主 streaming VAE 消费，必须先执行当前 suffix
并收齐 4 个新 observation，才能进行下一次 KV-cache update。从第二个转换起，cache/generation
与 action execution 真正重叠。

## 5. 配置和消融

主要环境变量：

- `LINGBOT_VA_CONSTRAINT_MODE=NONE|RTC|FBFM`
- `LINGBOT_VA_RTC_DELAY`、`LINGBOT_VA_RTC_EXECUTION_HORIZON`
- `LINGBOT_VA_RTC_ATTENTION_SCHEDULE=LINEAR|EXP`
- `LINGBOT_VA_FEEDBACK_OBS_PER_STATE=4`
- `LINGBOT_VA_FEEDBACK_LIVE=0|1`

`LINGBOT_VA_FEEDBACK_LIVE=0` 只用于 `FBFM-static` 诊断消融：反馈仍被接收、编码并持久化，但不
更新正在运行的 context。它不是第四种约束模式。

四组 Robotwin 命令：

```bash
script/run_constraint_ablation.sh NONE 1
script/run_constraint_ablation.sh RTC 1
script/run_constraint_ablation.sh FBFM-static 1
script/run_constraint_ablation.sh FBFM 1
```

RoboTwin 虽使用 llvmpipe 渲染，但 CuRobo planner 导入仍需要可见 CUDA。脚本要求策略 server 和
客户端使用不同物理 GPU（默认 client 为 server 的下一张卡），避免客户端约 3GiB CUDA context
挤占 FSDP unshard 峰值显存。

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
时序与 window cutoff、静态消融、solver 边界队列、双 WebSocket 并发和分布式 wrapper 队列路径。

## 7. 论文关系

- RTC（*Real-Time Execution of Action Chunking Flow Policies*）定义 previous-action 的异步衔接、
  `H/d/s` 与 soft mask；这里直接复用其 action 约束。
- LingBot-VA（*Causal World Modeling for Robot Control*）提供因果 video-action、KV cache 与真实
  observation 重新落地的系统框架。
- DreamZero（*World Action Models are Zero-shot Policies*）同样说明 WAM 的视频未来与 inverse
  dynamics 关系，并展示真正实时化依赖系统级流水线。

因此，FBFM 不是替换 RTC：它等于“完整 RTC action constraint + chunk 内实时 state constraint”。
三篇论文的结构化阅读笔记位于 `/mnt/project_eai_hs/zrm2/paper_notes/rtc_lingbotva_dreamzero.md`。
