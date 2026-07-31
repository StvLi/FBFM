# LingBot-VA 接入线的 FBFM 修复方案

## 文档目的

这份文档用于说明：当前 FBFM 在 LingBot-VA 上的接入实现，应该如何修改，才能更接近我们希望的方法语义。

最核心的边界是：

- 不重写基模 LingBot-VA 的原始 rollout / eval 语义。
- 不把基模原始推理行为本身当成 bug 去修。
- 只把 FBFM 当作一个推理时插件层来修正。

对当前 LingBot-VA 这条线，我们采用如下原则：

- feedback 的编码与消费方式，应当服从基模真实的 rollout 模式。
- 对 LingBot-VA 来说，这意味着优先遵守它原本的 causal / streaming rollout 语义，而不是把 feedback 当成独立静态窗口做无状态编码。


## 目标方法语义


我们真正想实现的是：

1. 当前 chunk 执行过程中，持续获得真实观测。
2. 这些真实观测被编码成与下一轮 latent 生成时间槽位对齐的状态反馈。
3. 下一轮 chunk 的生成过程中，flow matching 的速度场被 inference-time 的伪逆引导项修正。
4. 状态反馈和动作反馈都可以参与约束。
5. 如果状态反馈整条链不起作用，只剩动作前缀约束，那方法就退化成 RTC。

所以实现上必须保证 3 个不变量：

- Jacobian 项必须是真的，不能是 detach 之后的伪近似。
- 执行期间收集到的 feedback 必须能活到真正生成下一 chunk 的那一步。
- feedback 的对齐基准必须是 latent 生成时间轴，而不是只看 raw obs 的打包方式。


## 非目标

这份方案不会做下面这些事：

- 重设计 LingBot-VA 原始 rollout 逻辑；
- 重训基模；
- 在关闭 FBFM 时改变 LingBot-VA 原始 eval 行为；
- 强行要求所有基模都采用同一种 feedback 编码策略。

这里讨论的只是：**当前 LingBot-VA 接入线上的 FBFM 插件层应该怎么修**。


## 当前存在的问题

### 1. 伪逆引导的 Jacobian 项当前是坏的

在 `wan_va/lingbot_va_bridge.py` 里，当前 wrapped scheduler 的写法是：

- 先算 `v_t = original_denoise_step_partial(x_t)`
- 再做 `x_t.requires_grad_(True)`

这会导致 `v_t` 对 `x_t` 的依赖没有进入计算图。

结果就是，后面的 correction 不再真正包含 `\hat{X}^1` 对 `X_t^\tau` 的导数项。实现效果会更接近“按残差 nudging 一把”，而不是你们方法里真正需要的伪逆引导。

为什么这必须修：

- Jacobian 项是方法定义的一部分。
- 这个地方如果不对，后面所有反馈逻辑即使接通了，也不是你们想要的方法。
- 这属于 FBFM 插件层问题，不是基模问题。


### 2. 执行过程中收集到的 feedback 会在真正推理前被覆盖掉

当前 server 的逻辑是：

- feedback 到来时，先 append 到 `self.prev_chunk_left_over`
- 但真正推下一 chunk 之前，又重新 new 了一次 `PrevChunkAdapter`

这样会造成语义断裂：

- 执行期间 feedback 的确收集了；
- 但到了真正需要消费 feedback 的那一步，存 feedback 的对象可能已经被重建了。

为什么这必须修：

- FBFM 的关键就在于把异步执行期间拿到的真实反馈带入下一轮 guided generation；
- 如果在生成前把 feedback 覆盖掉，那整个插件层最核心的信号就丢了；
- 这同样是接入层问题，不是基模原生逻辑的问题。


### 3. 动作反馈现在还不是“真正的 leftover / prefix”

当前实现中，`last_action` 保存的是整块上一轮动作，然后初始化 `PrevChunkAdapter` 时直接把整块动作展开成 prefix 约束。

但按你们方法的语义，动作反馈不该只是“整块上一轮动作”。它应该体现异步 delay 语义下、与下一轮生成对齐的动作信息。对当前 LingBot-VA 线来说，更自然的理解是：

- 它应该是 leftover / prefix 型动作约束；
- 而不是把整个 previous chunk 原封不动拿来当 prefix。

为什么这必须修：

- RTC 风格的动作约束，只有在 prefix 时间对齐正确时才有意义；
- 现在把整块前一 chunk 动作全塞进去，不等于传递了“当前应该约束的动作前缀”；
- `inference_delay` 现在虽然一路传进来了，但还没有真正参与 slicing / 对齐逻辑，说明这条语义链没有落地。


### 4. 状态反馈当前是按 raw obs 打包，不是按 latent slot 对齐

当前 client/server 交互更像是这样：

- 发最近 4 帧 raw obs；
- VAE 编码；
- 往 state buffer 里逐个 append latent state；
- buffer 满了就截断。

这只是一个粗糙启发式，不是方法上严格成立的对齐方式。

你们真正想要的语义应该更强：

- state feedback 最终要对齐到“下一轮生成时使用的 latent state slot”；
- 而不是“先打一个固定 raw obs 窗口，再看编码后能塞几个 state slot”。

为什么这必须修：

- 你们的引导对象是在 latent state space；
- 所以主对齐基准必须是 latent 时间轴；
- 如果这点不成立，就算 state feedback 接上了，也可能天然是错位的。


### 5. feedback 编码现在把 streaming 语义和重叠窗口重编码混在了一起

对于 LingBot-VA，我们已经同意应该采用 A：

- feedback 编码应该遵守基模 rollout 的 causal / streaming 语义。

但当前实现混合了两套互相冲突的东西：

- 一方面复用了 streaming VAE cache；
- 另一方面又反复拿重叠的 “latest-N-frame” 窗口去编码。

这并不是一个干净的 causal stream。

为什么这必须修：

- 如果 feedback 属于主 rollout 流，那每个观测都应该单调推进这条流；
- 如果 feedback 是独立测量流，那就不该和主 streaming cache 混用；
- 现在这种“复用 cache + 重叠滑窗”的写法，会让“真实反馈 latent”到底代表什么变得不清楚。


## 设计决策

### 决策 A：遵守基模 rollout 模式

对 LingBot-VA 来说，FBFM 应该适配它原本的 rollout 方式，而不是反过来要求基模去适应插件。

更具体地说：

- LingBot-VA 本身就是 causal rollout + KV cache + streaming video encoding；
- 所以 FBFM 的 feedback 语义，应当建立在这套真实 rollout 方式之上。

这里要强调：

- 这不等于“当前代码已经正确实现了 A”；
- 这只是说明，修复时应该朝 A 的方向收敛，而不是改成无状态局部窗口编码。


### 决策 B：保持 FBFM 是插件层

修复时要保持职责清晰：

- 基模继续负责自己的 rollout；
- feedback 的收集、对齐、引导注入由 FBFM 层负责。

这样做的原因是：

- 将来同一个实时推理插件要适配不同基模；
- 不同基模的 rollout 方式可能不同；
- 插件层不能把某一种 rollout 假设写死成唯一真理。


### 决策 C：优先保留分步式 FBFM，而不是强行联合重写

你们的理论公式是在联合变量 `X=[Z,A]` 上写的，这没有问题。

但当前 LingBot-VA 工程结构天然是两段：

- 先 video generation loop；
- 再 action generation loop。

为了不破坏基模原始 eval 语义，当前阶段更稳妥的修法是：

- 对 video latent generation 做 state feedback 引导；
- 对 action generation 做 action feedback / prefix 引导；
- 在解释层面仍然把两者看作联合变量 `X=[Z,A]` 的分步实现。

这比强行把 LingBot-VA 重写成一个全联合 scheduler 路径更安全。


## 具体准备怎么改

### 修复 1：先把 scheduler 的梯度链修正

准备改什么：

- 保证 `x_t` 在 denoiser 前向前就进入 autograd；
- 在有梯度追踪的上下文里重新计算 `v_t`；
- 确认 correction 真实反映 `\hat{X}^1` 对 `X_t^\tau` 的导数。

为什么先改这个：

- 这是 FBFM 数学上最核心的一步；
- 如果这一步不对，后面反馈链路全接通也无法说明“你们方法实现了”；
- 这是最应该优先修的地方。


### 修复 2：把 feedback 的持久化和每次 infer 时的 adapter 构造拆开

准备改什么：

- 不允许执行期间收集到的 feedback 在真正生成前被 new adapter 覆盖掉；
- feedback 的累积状态要有稳定存储；
- 如果确实需要为了 shape/layout 重建 adapter，也应该从持久化 feedback 状态重建，而不是从空状态重建。

为什么这样改：

- FBFM 的关键资源就是“执行期间拿到的真实反馈”；
- 如果这条链活不到真正 denoising 的那一步，方法语义就断了。


### 修复 3：把动作约束从“整块上一轮动作”改成“真正的 leftover / prefix”

准备改什么：

- 不再直接把整块 `last_action` 当 prefix；
- 用 async delay 语义决定，当前轮到底该约束上一轮动作里的哪一部分；
- 让 `inference_delay` 真正参与 slicing 和对齐。

为什么这样改：

- 你们方法里的 action feedback 本质上是时间对齐后的动作上下文；
- 不是“整个 previous chunk 动作全给你”；
- 如果这条语义不修，action guidance 永远是错位的。


### 修复 4：把 state feedback 改成按 latent slot 需求对齐

准备改什么：

- 显式建立：
  - 真实观测到达时间；
  - 编码后 latent 时间；
  - 下一 chunk latent slot；
 这三者之间的映射；
- 不再依赖“固定 raw obs 窗口 + 塞满即截断”作为主对齐机制；
- 下一轮 state feedback 的消费量应由 latent generation schedule 决定。

为什么这样改：

- 引导发生在 latent state space；
- 所以 latent slot 对齐必须是主语义；
- raw obs 的打包方式只能是实现细节，不能反过来主导方法定义。


### 修复 5：把 feedback 编码的流语义写清楚并落地

准备改什么：

- 明确 feedback 编码到底是：
  - 主 rollout 的 streaming encoder state；
  - 还是独立 feedback encoder state；
- 对 LingBot-VA 来说，优先保留 causal rollout 语义；
- 但必须保证实现一致：
  - 要么严格单调推进 streaming；
  - 要么 feedback 独立编码并隔离 cache；
- 避免当前这种“重叠滑窗 + 共用 streaming cache”的混合状态。

为什么这样改：

- 现在最大的问题之一，不是“编码出来没有数值”，而是“编码出来的 latent 到底算不算真实反馈”；
- 如果这个定义本身不稳定，后面 state guidance 的解释会非常困难。


## 推荐的 debug / 修复顺序

建议按下面顺序来：

1. 修 scheduler 的 Jacobian 链路
2. 修 feedback 生命周期 / 覆盖问题
3. 修 action leftover 对齐
4. 修 state slot 对齐
5. 修 feedback encoder 的 cache 语义

为什么这个顺序合适：

- 前两项决定 FBFM 是否“存在”为一个真正工作的引导机制；
- 中间两项决定它是否真的符合 async rollout 语义；
- 最后一项决定反馈信号本身是否可信。


## 哪些东西不该改

当 FBFM 关闭时，应该尽可能满足：

- LingBot-VA 的原始 eval / rollout 行为保持不变；
- FBFM 层退化成 no-op，或者至少语义上等价于 no-op；
- 不应该因为引入插件层，就改变基模原始的推理方式。

这是当前接入线必须坚持的硬边界。


## 修完之后理想上应该达到什么状态

修完之后，LingBot-VA 接入线应该满足下面这个解释：

- LingBot-VA 继续按它原本的 causal rollout 和 async chunked control 去跑；
- FBFM 只作为推理时闭环修正插件存在；
- state feedback 对齐到 latent rollout 的时间槽位；
- action feedback 对齐到 async delay 语义下的 leftover / prefix；
- guidance 算子在数学上是有意义的，不是 detach 之后的残差启发式。

到那个时候，才可以比较诚实地说：

- 这是一个叠加在 LingBot-VA 之上的 FBFM 插件；
- 而不是一个带 feedback 接口、但方法语义还没真正落地的改造版 eval 路径。


## 执行 checklist

### 当前 goal

当前 goal 定义为：

- 在**不使用 GPU**、**不打断现有训练/推理任务**的前提下，
- 先把 FBFM 接入层的问题定位链路跑通，
- 再按最小修改原则开始修复 FBFM 插件层，
- 不改 LingBot-VA 关闭 FBFM 时的原始 eval 行为。

当前阶段的硬边界：

- 不启动任何新的 GPU 进程；
- 不 kill、不暂停、不抢占现有 GPU 任务；
- 不做训练；
- 不跑完整 RoboTwin 推理；
- 只做 CPU 上的静态检查、局部张量路径检查、日志插桩准备和代码修复。


### Phase 0：执行前检查

- 确认当前机器上已有 GPU 任务正在运行，仅记录、不干预。
- 确认本轮所有操作限定在文档、代码静态分析、CPU 命令和轻量单测范围内。
- 确认需要观察的核心文件：
  - `wam/lingbot-va/wan_va/lingbot_va_bridge.py`
  - `wam/lingbot-va/wan_va/wan_va_server.py`
  - `wam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py`
  - `fbfm/policies/fbfm/modeling_rtc_fbfm.py`

验收信号：

- 没有新增 GPU 负载；
- 没有影响现有运行中的任务；
- 后续调试范围已经收敛到 FBFM 插件层。


### Phase 1：先做静态链路核对

- 画清楚 3 条链路：
  - state feedback 收集链路；
  - action leftover / prefix 链路；
  - scheduler guidance 梯度链路。
- 明确 `prev_chunk_left_over` 在以下阶段的生命周期：
  - reset
  - feedback
  - compute_kv_cache
  - infer one chunk
- 明确 `last_action` 的来源、存储格式、消费位置。
- 明确 `state_num`、`frame_chunk_size`、`action_per_frame`、`INFER_DELAY_STEPS` 的时间语义。

验收信号：

- 每条链路都能用具体函数和变量名描述清楚；
- 能明确指出每个核心状态对象在哪里创建、更新、覆盖和消费；
- 对“什么时候 FBFM 退化成 RTC / None”能给出代码级判据。


### Phase 2：先修 scheduler 的 Jacobian 链路

- 检查 `x_t.requires_grad_(True)` 的位置是否在 denoiser 前向之前。
- 检查 `v_t` 是否真实依赖于可求导的 `x_t`。
- 检查 `correction` 的 autograd 路径是否包含 `x_t -> v_t -> x1_t`。
- 增加最小级别的 CPU/单元测试，验证：
  - 有 guidance 时 `correction` 不是恒零；
  - 改变 `constrained_y` 或 `weights` 时 `correction` 会变化；
  - 关闭 guidance 时 wrapper 退化为原始 scheduler 行为。

验收信号：

- 梯度图不再在 `v_t` 前断掉；
- `correction` 对反馈目标敏感；
- 关闭 guidance 时输出路径与原始 LingBot-VA 保持一致语义。


### Phase 3：修 feedback 生命周期

- 把 feedback 的累积状态从“临时 adapter 对象”里解耦出来。
- 保证执行期间 append 的 state feedback 不会在真正生成前被覆盖。
- 如果确实需要重建 `PrevChunkAdapter`，则从持久化反馈状态重建，而不是从空状态重建。
- 增加日志点，仅打印轻量状态摘要：
  - `state_constrained_num`
  - `action_constrained_num`
  - adapter 重建前后的 buffer 长度

验收信号：

- feedback 到来后，下一次真正 `_infer` 前后 buffer 内容连续；
- 不再出现“append 了 feedback，但 infer 前对象被清空”的情况；
- 同一轮异步执行期间收集到的反馈能稳定存活到生成步骤。


### Phase 4：修 action leftover / prefix 对齐

- 定义 `last_action` 在当前实现里的正确语义：
  - 不是整块 previous chunk；
  - 而是与 async delay 对齐后的 leftover / prefix。
- 让 `inference_delay` 真正参与 slicing。
- 明确 prefix 是按：
  - 已执行动作；
  - 未执行 leftover；
  - 或与状态时间点一一对应的动作片段；
 选定一种并在代码中贯彻。
- 给 adapter 增加 CPU 级输入输出形状和索引校验。

验收信号：

- `action_constrained_num` 与 async delay 语义一致；
- prefix 不再等于整块上一轮 action；
- action guidance 的时间语义和文档定义一致。


### Phase 5：修 state feedback 对齐

- 明确 raw obs 时间轴、latent 时间轴、next chunk latent slot 之间的映射。
- 不再默认“固定上传最近 4 帧 obs == 正确反馈窗口”。
- 如果 LingBot-VA 真实 rollout 语义要求 streaming 编码，则反馈应按真实增量推进。
- 如果当前 client 仍发送窗口，则 server 端必须显式做 slot 对齐，而不是简单 append 后截断。

验收信号：

- 能说明每一个 state slot 对应的是哪一段真实观测；
- `state_constrained_num` 的增长逻辑与 latent slot 需求一致；
- state guidance 不再依赖隐式截断碰运气。


### Phase 6：修 feedback encoder cache 语义

- 明确 feedback 编码到底属于：
  - 主 rollout streaming encoder；
  - 还是独立 feedback encoder。
- 如果属于主 rollout：
  - 必须保证输入是单调推进的，不允许重叠窗口反复污染同一 cache。
- 如果属于独立 feedback encoder：
  - 则应显式隔离 cache，不与主 rollout 编码状态混用。
- 用 CPU 级 mock / shape test 验证同一输入重复编码时行为是否符合预期语义。

验收信号：

- feedback latent 的语义稳定，可解释；
- 不再出现“重叠窗口 + streaming cache 共用”的混合状态；
- 可以明确回答“当前反馈 latent 是否属于真实 rollout 状态估计”。


### Phase 7：回归约束

- 验证 FBFM 关闭时：
  - 不改变 LingBot-VA 原始 eval 行为语义；
  - wrapper 退化成 no-op 或等价 no-op；
  - 不引入额外 side effect。
- 验证 FBFM 开启时：
  - state guidance 真正参与 video 生成；
  - action guidance 真正参与 action 生成；
  - 两者都来自时间对齐后的反馈，而不是空 buffer 或错误 prefix。

验收信号：

- FBFM off：基模原始行为保持稳定；
- FBFM on：能从日志和局部张量摘要中看见 guidance 生效；
- 退化到 RTC / None 的条件可被显式判定，而不是靠猜。


### 当前执行策略

在 GPU 空出来之前，先执行到这里：

1. 完成静态链路核对；
2. 完成 Jacobian 链路修复设计；
3. 完成 feedback 生命周期修复设计；
4. 完成 action/state 对齐策略定稿；
5. 把需要的最小日志点和最小单测设计好。

GPU 空出来之后，再执行：

1. 局部前向验证；
2. 单步 guidance 数值检查；
3. 短 episode 真实链路回归。

这样可以把“占 GPU 的验证步骤”推迟到最后，避免干扰当前正在运行的重要任务。


## Phase 1 静态链路核对结果（CPU-only）

这一节记录当前已经完成的静态核对结果。这里不涉及 GPU，不依赖实际 rollout，只基于代码路径和变量生命周期做结论。


### 1. state feedback 链路

当前 state feedback 的主路径是：

1. client 在动作执行过程中，每 `4` 个低层 action 取一次观测；
2. 将 `full_obs_list[-4:]` 通过 `feedback=True` 发给 server；
3. server 在 `_feedback()` 中调用 `_encode_obs()`；
4. 编码结果通过 `self.prev_chunk_left_over.append_new_state(...)` 进入状态缓存；
5. video generation loop 在 `scheduler.step(...)` 时消费：
   - `get_constrained_states()`
   - `get_state_prefix_weights()`

静态结论：

- `feedback` 的消费对象是 `self.prev_chunk_left_over`；
- 但这个对象在真正 `_infer()` 前会再次重建；
- 因此“执行期间累积的 state feedback”与“真正消费 feedback 的 generation adapter”当前不是稳定绑定的。

更具体地说：

- `_reset()` 会删除 `prev_chunk_left_over`；
- `feedback=True` 路径会直接对当前对象 append；
- `compute_kv_cache` 之后、以及真正 `Infer One Chunk` 之前，又都会重建 `PrevChunkAdapter`；
- 所以如果不额外做持久化或重建迁移，feedback 很容易在消费前丢失。

这说明：

- 当前实现里，state feedback 已经“接到了 server”，
- 但还没有“稳定接到下一轮 guided generation”。


### 2. action leftover / prefix 链路

当前 action 约束链路的主路径是：

1. `_infer()` 完成后，`actions = self.postprocess_action(actions)`；
2. `self.last_action = action` 保存的是后处理之后的动作块；
3. 下次构造 `PrevChunkAdapter` 时，把 `prev_actions=self.last_action` 传进去；
4. adapter 将整块 previous action 展平成 `(F*N, D)`；
5. `action_constrained_num` 直接取整块动作长度；
6. action loop 在 `scheduler.step(...)` 时消费：
   - `get_constrained_actions()`
   - `get_action_prefix_weights()`

静态结论：

- 当前 prefix 语义不是“leftover 动作前缀”；
- 而是“整块上一轮动作”；
- `inference_delay` 虽然被一路传入，但没有真正参与 slicing / 对齐。

这意味着：

- 当前 action guidance 的时间语义还不是你们方法里想要的 async-leftover 语义；
- 它更像是“上一轮完整 action block 的整体约束”。

补充说明：

- client 在 `compute_kv_cache=True` 时还会把 `state=action` 发给 server；
- 这条链主要用于 base model 的 action cache 构造；
- 它和 `self.last_action` 这条 FBFM prefix 语义链并不是同一个概念。


### 3. scheduler guidance 梯度链路

当前 wrapped scheduler 的关键路径是：

1. `x_t = x_t.clone().detach()`
2. `v_t = original_denoise_step_partial(x_t)`
3. `x_t.requires_grad_(True)`
4. `x1_t = x_t - sigma * v_t`
5. `correction = autograd.grad(x1_t, x_t, ...)`

静态结论：

- `v_t` 是在 `x_t.requires_grad_(True)` 之前算出来的；
- 所以 `v_t` 对 `x_t` 的依赖没有进 autograd 图；
- 后面的 correction 无法真实表达 `\partial \hat{X}^1 / \partial X_t^\tau`。

这说明：

- 当前 guidance 在数学上不是完整的 FBFM / PiGDM 形式；
- 即使反馈链路其他部分都修通，这里不修，方法本体仍然不成立。


### 4. `prev_chunk_left_over` 的生命周期结论

当前这个对象承担了两种职责：

- 存放 action prefix；
- 存放 state feedback。

但它的生命周期目前是混杂的：

- reset 时清空；
- first infer 时初始化；
- feedback 时原地 append；
- `compute_kv_cache` 后可能重建；
- 真正 infer one chunk 前也会重建。

静态结论：

- 这是当前最明显的职责耦合点；
- action prefix 的构造需求和 state feedback 的累积需求被绑在同一个短生命周期对象上；
- 后续修复时，必须把“反馈累积状态”和“每轮生成用的 adapter 视图”拆开看。


### 5. 当前阶段最硬的静态结论

在不考虑开关是否开启的前提下，当前实现里最硬的 3 个静态问题已经可以确定：

1. scheduler 的 Jacobian 链路是坏的；
2. feedback 生命周期不稳定，存在消费前被覆盖的风险；
3. action prefix 还不是 async-leftover 语义，而是整块 previous action 语义。

这 3 个问题不需要 GPU 就可以确认，也足以决定后续修复优先级。


## 当前已执行进展（CPU-only）

这一节记录截至当前已经实际落地的修改，以及哪些内容已经有 CPU 级验证支撑。


### 已完成 1：修 scheduler 的 Jacobian 链路

已做修改：

- 将 `x_t.requires_grad_(True)` 前移到 denoiser 前向之前；
- 保证 `v_t` 在可求导的 `x_t` 上计算；
- 保留 no-guidance 路径的原始 scheduler 语义。

当前 CPU 验证结果：

- wrapper 在 no-guidance 情况下可退化为 base scheduler 等价行为；
- 改变 `constrained_y` 或 `weights` 时，wrapper 输出会变化；
- 当前最小单测已通过。


### 已完成 2：把 state feedback 的持久化从临时 adapter 中解耦

已做修改：

- 增加了独立的 `FeedbackStateBuffer`；
- feedback 不再直接只依赖 `prev_chunk_left_over` 这个短生命周期对象；
- `PrevChunkAdapter` 构造时从持久化 feedback buffer 导出最近的状态约束。

当前效果：

- feedback 累积状态与“每轮推理时构造的 adapter 视图”已经分离；
- infer 前重建 adapter 时，不再天然等于丢失 feedback。

说明：

- 这解决的是生命周期问题；
- 不等于 state feedback 的时间对齐已经完全正确。


### 已完成 3：把 action prefix 改成按 `inference_delay` 取 tail

已做修改：

- `PrevChunkAdapter` 不再默认使用整块 previous action；
- 当 `inference_delay > 0` 时，改为取上一轮动作块尾部的 `inference_delay` 个 action step 作为 prefix；
- `action_constrained_num` 因而与 async delay 语义绑定。

当前 CPU 验证结果：

- 最小单测验证了 prefix 取的是 tail，而不是整块 previous action；
- `action_constrained_num` 与 tail 长度一致。

说明：

- 这里采用的当前实现策略是：**把 leftover 解释为 previous chunk 的未执行尾部，并将其映射成下一轮的前缀约束**；
- 这已经比原来“整块 previous action 直接拿来约束”更接近方法语义。


### 已完成 4：增加最小 CPU 单测

已新增测试覆盖：

- wrapper no-guidance 退化行为；
- wrapper guidance 对 `constrained_y` / `weights` 的敏感性；
- `FeedbackStateBuffer` 的最近状态导出逻辑；
- `PrevChunkAdapter` 的 action tail prefix 语义。

当前结果：

- `python -m pytest tests/test_fbfm_bridge.py -q`
- `4 passed`

并且额外通过了：

- `python -m py_compile` 对本轮修改文件的语法检查。


## 当前仍未彻底完成的部分

下面这些问题还没有被“完全解决”，只是已经从结构上向正确方向推进：

### 1. state feedback 的最终时间对齐

当前已经做到：

- feedback 有独立持久化状态；
- adapter 构造时会显式导出最近的 `state_num` 个状态。

但还没有最终回答：

- raw obs 时间轴与 latent slot 时间轴的精确映射；
- 当前 client 固定发送窗口时，server 是否应该只消费最新增量，还是消费整个窗口编码结果中的某一部分。

这部分仍需要 GPU 空出来之后，结合真实 encoder 输出形状和 rollout 观察来确认。


### 2. feedback encoder cache 语义

当前还没有最终定稿：

- feedback 编码到底是否应当严格属于主 rollout streaming state；
- 还是应该在当前 client 滑窗输入条件下，退而求其次地使用隔离 feedback encoder state。

目前代码结构已经为后续修正生命周期和状态导出打好了基础，但 cache 语义本身还需要进一步验证。


### 3. FBFM on/off 的端到端回归

当前只完成了：

- 桥接层 CPU 单测；
- scheduler 数学路径的局部验证。

还没有做：

- FBFM off 时对 LingBot-VA 原始 eval 行为的端到端回归；
- FBFM on 时对 video/action guidance 的真实 rollout 验证。

这部分必须等 GPU 空出来之后再做。


## Phase 5 / 6 当前已选定的实现策略

在不使用 GPU 的前提下，当前已经把 state feedback 对齐和 feedback encoder cache 语义收敛到下面这套实现策略：

### 策略 1：client 的 feedback 窗口只提供“最新真实观测”的传输通道

虽然 client 当前发送的是 `latest-4` 窗口，但在 server 端，我们不再把整个重叠窗口重复当作新的 feedback 序列去编码。

当前采取的语义是：

- `compute_kv_cache=True` 时传来的 4 帧窗口，用作新一轮 feedback 流的初始种子；
- 后续 `feedback=True` 调用时，只消费窗口中的**最新一帧**，把它视为“本次真实新增观测”。

这样做的原因是：

- client 的窗口是重叠的；
- 如果每次把整个窗口重新喂给 feedback encoder，就会人为重复历史观测；
- 这不符合 causal rollout 语义。


### 策略 2：feedback 使用独立的 streaming wrapper，但共享基模 VAE 权重

当前实现里：

- 主 rollout 继续使用原有 `self.streaming_vae`；
- feedback 路径新增独立的 `feedback_streaming_vae`；
- 两者共享同一个底层 VAE 权重，但各自维护自己的 cache 状态。

这样做的原因是：

- 我们仍然遵守 A：feedback 属于 causal / streaming 编码语义；
- 但不能让 feedback 的滑窗输入去污染主 rollout 的 encoder cache；
- 因此“共享权重、隔离 cache”是当前最稳妥的折中。

这意味着：

- feedback 仍然是流式编码，不是无状态局部编码；
- 但它不会反向污染基模主 rollout 的内部缓存。


### 策略 3：4 个 feedback 观测对应 1 个 state slot

当前实现采用：

- `obs_per_state = 4`
- 每累计 4 个 feedback 观测，产出 1 个 state feedback slot

依据是：

- LingBot-VA 论文与数据处理都明确有时间下采样因子 `4`；
- 当前 RoboTwin client 也是每 4 个低层 action 采样一次观测；
- 所以在当前接入线上，先按“4 个 feedback obs -> 1 个 latent state slot”实现，是最一致也最保守的选择。

当前代码中，这个映射已经由 `FeedbackSlotTracker` 固定下来。


### 策略 4：state buffer 只存储按 slot 语义产出的状态

当前实现不再是：

- 每次 `_encode_obs(latest-4)` 后把所有 latent frame 直接 append，满了就截断。

而是：

- feedback 观测单调推进；
- 只有当累计观测数跨过一个 slot 边界时，才把最新状态写入 `FeedbackStateBuffer`。

这样做的直接好处是：

- `state_constrained_num` 的增长不再依赖“当前窗口编码出了几个 latent frame”；
- 而是依赖显式定义好的 slot 对齐语义。


### 当前 CPU 验证结果

当前已经通过 CPU 级验证的内容包括：

- `FeedbackSlotTracker` 每 4 个观测发出 1 个 slot；
- `FeedbackStateBuffer` 能稳定导出最近状态；
- scheduler guidance 数学路径正常；
- action tail prefix 语义正常。

当前还没有完成的验证包括：

- 真实 LingBot-VA feedback encoder 在 GPU 上对单帧流式输入时的 latent 输出形状；
- 当前 `4 obs -> 1 slot` 是否和真实 encoder 输出完全吻合；
- FBFM on/off 在真实 rollout 中的端到端回归。


## 当前已完成的语义升级：从 recent-prefix 到 slot-indexed state guidance

在这一步之前，state feedback 虽然已经从“滑窗截断启发式”推进到了“按反馈节奏累计 recent states”，但本质上仍然属于：

- recent states
- prefix fill
- prefix mask

也就是说，当时的实现更接近：

- 取最近的若干 ground-truth state；
- 把它们默认塞进 chunk 前几个 slot；
- 用前缀型 `state_prefix_weights` 表示可用范围。

这还不是最终想要的 FBFM 语义。

### 本轮升级之后，state feedback 的内部表示已经改为：

1. `constrained_y_state` 按 next latent chunk 的 slot 对齐存储；
2. `W_state` 作为显式的 slot-aligned 0/1 availability mask 存储；
3. adapter 优先使用显式 `state_mask`，而不是自动退回“前 n 个有效”的 prefix mask。

换句话说，现在已经不再是：

- “最近几个 state 直接当前缀”

而是：

- “当前 next latent chunk 的 slot 0 / slot 1 / ... 各自是否已有 ground-truth；
- 如果有，则其对应 latent 是什么；
- 如果没有，则该 slot 的 mask 为 0。”


### 这一步具体意味着什么

当前 state guidance 语义已经从：

- recent-state prefix

升级成：

- next latent chunk 上的 slot-indexed `constrained_y + W_state`

这与当前方法定义更一致，因为：

- 引导发生在 video generation loop 的 latent chunk 上；
- 因而 state feedback 也应当按该 latent chunk 的时间槽位组织；
- 而不是仅仅保留一个“最近状态列表”。


### 当前 CPU 验证补充结果

本轮新增并通过的 CPU 验证包括：

- `SlotAlignedStateBuffer` 的状态与 mask 导出行为；
- `PrevChunkAdapter` 在提供显式 `prev_state_mask` 时，会优先使用 slot-aligned mask，而不是 prefix mask。

当前测试结果：

- `python -m pytest tests/test_fbfm_bridge.py -q`
- `7 passed`

这说明在纯 CPU 级别上，state guidance 的内部表示已经完成了从 prefix 语义到 slot-aligned 语义的升级。
