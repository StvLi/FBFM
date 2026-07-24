# LingBot-VA + FBFM 因果链辅助实验设计

状态：实验方案，待实现与预注册

适用范围：LingBot-VA + FBFM 在 RoboTwin 中的确定性伪异步评测

不适用范围：DreamZero、真实 wall-clock 异步吞吐比较

## 1. 实验目标

本实验不只验证 FBFM 是否改变了 action，而是验证下面的完整逻辑链：

```text
执行期间获得的真实反馈
        ↓
video generation loop 的 next-state prediction 更准确
        ↓
更准确的 predicted video KV cache 被 action generation loop 读取
        ↓
fresh action suffix 发生有益变化
        ↓
RoboTwin 局部任务进展与完整 episode 成功率提高
```

需要分别建立三条证据：

1. `Feedback -> Video`：真实反馈降低 next-state prediction error。
2. `Video -> Action`：action 的变化由 predicted video cache 介导。
3. `Action -> Success`：这种 action 变化提高任务表现。

只有三条证据同时成立，才支持“FBFM 通过反馈改善未来状态生成，进而改善动作和成功率”的完整表述。

## 2. LingBot-VA 中的实际信息路径

当前推理顺序是 video-first、action-second：

1. video generation loop 在每个伪 solver boundary 接收已经到达的 live feedback；
2. state constraint 只传入 video scheduler；
3. video loop 最后的 padded `t=0` cache-only 调用把生成视频写入 predicted KV cache；
4. action generation loop 随后运行，并通过共享 self-attention cache 读取生成视频；
5. action loop 不直接读取 feedback latent 或 state constraint。

对应代码边界：

- video generation loop：`wan_va/wan_va_server.py::_infer`；
- state constraint snapshot：video scheduler 调用之前；
- action generation loop：紧跟在 video loop 之后；
- predicted KV cache：`wan_va/modules/model.py::WanAttention.update_cache`，其中 `update_cache == 1` 的条目标记为 `is_pred`。

因此，LingBot-VA 最合适的中介变量不是 DreamZero 式 joint-VJP 的 action-coordinate gradient，而是：

```text
predicted video latent / predicted video KV cache
```

## 3. 固定协议与禁止改动项

辅助实验必须保持当前正式伪异步协议：

| 变量 | 固定值/规则 |
|---|---:|
| video latent chunk | 2 slots |
| actions per latent frame | 16 |
| action horizon `H` | 32 |
| inference delay `d` | 16 |
| execution horizon `s` | 16 |
| video scheduler | 25 numerical steps + 1 padded cache-only call |
| pseudo video solver budget | 26 grants |
| action scheduler | 原生 action schedule |
| observations per feedback latent | 4 |
| feedback observation interval | 每 4 个 simulator action steps一次 |
| active feedback slots per wave | 通常为第一个 video slot |

以下内容在组间必须完全一致：

- checkpoint、prompt 和模型权重；
- RoboTwin task、trial seed、初始 simulator state；
- real-history KV cache；
- video initial noise 和 action initial noise；
- video/action scheduler 与 CFG；
- committed action prefix、RTC action target 和 action mask；
- feedback 到达的虚拟 action step；
- 26 个 video solver grants 的顺序；
- episode step budget 和成功判定；
- 数值精度与 GPU 配置。

wall-clock latency 只作为资源指标，不参与任何方法变量、反馈可见性或 solver step 分配。

## 4. 实验组定义

### 4.1 RTC

- action constraint：开启；
- live state feedback：接收但 state mask 对 video loop 为零；
- action loop cache：RTC video loop 生成的 cache，记为 `C_R`。

RTC 是全部实验的直接基线。

### 4.2 FBFM

- action constraint：与 RTC 完全相同；
- live state feedback：使用当前 wave 的真实、时序对齐 feedback；
- action loop cache：FBFM video loop 生成的 cache，记为 `C_F`。

FBFM 是完整方法。

### 4.3 FBFM-Shuffled

- action constraint：与 RTC/FBFM 完全相同；
- state mask、state weight、反馈到达时刻：与 FBFM 完全相同；
- feedback target：替换为相同 task、相同 latent slot、不同 trial/wave 的 feedback latent；
- action loop cache：错误反馈引导后得到的 cache，记为 `C_S`。

Shuffled mapping 必须预先生成并固定：

- 只在相同 task 内置换；
- 保持相同 slot 和相近 action step；
- 使用 derangement，禁止样本匹配到自身；
- mapping 只由预注册随机种子决定；
- 不能根据 FBFM 结果重新选择 shuffled target。

该组用于排除“任意非零 guidance、额外计算或随机扰动都能提升”的解释。

### 4.4 FBFM-CacheCut

先完整运行真实反馈引导的 FBFM video loop并保存其预测与误差；进入 action loop 前，切断 video-to-action 中介路径：

1. 清除当前 predicted cache；
2. 恢复同一 wave 配对 RTC video loop 的 `C_R`；
3. 使用与 RTC 完全相同的 action initial noise 运行 action loop。

该组中：

- video prediction 应保持 FBFM 的改善；
- action loop 实际读取 RTC video cache；
- fresh action suffix 应退回 RTC。

该组用于检验改进后的 video prediction 是否是 action 收益的必要中介。

### 4.5 RTC+FBFMCache

这是只用于机制实验的互逆 cache transplant：

1. 外层 video 条件使用 RTC；
2. action loop 前清除 RTC predicted cache；
3. 安装同一 wave 配对 FBFM 的 `C_F`；
4. 使用与 FBFM 完全相同的 action initial noise 运行 action loop。

该组用于检验 FBFM video cache 是否足以复现 FBFM action。它不需要作为正式方法跑完整 50-task benchmark。

## 5. 模块 A：验证 Feedback -> Video

### 5.1 验证问题

真实且时序对齐的 feedback 是否让 video generation loop 的 next-state endpoint 更准确？这种改善是否优于错误反馈？

### 5.2 使用实验组

- RTC；
- FBFM；
- FBFM-Shuffled。

三组必须从同一个 wave launch state、相同 video noise 和相同伪异步 trace 开始。

### 5.3 配对 replay 过程

每个有效 wave 执行一次 committed 16-action suffix，同时记录：

- wave 启动前的 real-history KV cache 标识；
- video noise generator state；
- 16 个实际执行动作；
- 四个 feedback observations；
- feedback window 形成的 action step；
- 每个 simulator step 发放的 solver grant；
- feedback constraint version；
- wave 结束时的 simulator state。

随后按相同虚拟时间顺序重放 video loop：

```text
同一 launch state + 同一 video noise + 同一 solver grant trace
    ├── RTC：state mask = 0
    ├── FBFM：state target = 当前真实 feedback
    └── Shuffled：state target = 预注册错误 feedback
```

重放可以顺序进行，不要求三条 GPU inference 真正同时运行。伪异步公平性由反馈可见 step 和 solver grant 顺序决定。

### 5.4 Endpoint 定义

在 video solver step 的 sigma 坐标下：

```text
z_endpoint = z_sigma - sigma * v_video
```

对所有组都使用同一个真实 feedback latent 计算评估误差；RTC 和 Shuffled 不能把这个真实 target 用于 guidance。

### 5.5 已观察 next-state slot 指标

第一个 video slot 对应本 wave 执行期间已经形成的真实 feedback latent：

```text
E_observed = ||z_endpoint[slot0] - z_feedback[slot0]||_2
```

必须记录：

- feedback 可见前最后一个 solver step 的误差；
- feedback 首次可见 step 的 guidance 前误差；
- feedback 首次可见 step 的 guidance 后 endpoint 误差；
- 后续每个 video numerical step 的误差；
- final video sample 的 slot-0 error；
- state mask nonzero、constraint version、correction norm 和 guidance weight。

主要指标：

- slot-0 normalized latent MSE；
- slot-0 latent cosine similarity。

该指标证明 feedback 确实修正了与当前环境相对应的 next-state prediction。因为 slot 0 被直接约束，它是必要的机制验收指标，但不能单独证明对未观察未来的泛化。

### 5.6 未观察 future slot 指标

第二个 video slot 在当前 video loop 中没有被真实 feedback 直接约束。执行新生成的 fresh 16-action suffix后，再收集四个 observations，并用与生产路径相同的 causal VAE 编码真实 slot 1：

```text
E_future = ||z_final[slot1] - z_realized_future[slot1]||_2
```

报告：

- slot-1 normalized latent MSE；
- slot-1 latent cosine similarity；
- 解码画面的 LPIPS；
- 解码画面的 PSNR/SSIM。

slot-1 指标是更强的辅助证据，但不是 cache 中介成立的必要条件：action loop 可以直接受更准确的 slot 0 cache 影响。

### 5.7 模块 A 的判据

必要结果：

```text
E_observed(FBFM) < E_observed(RTC)
E_observed(FBFM) < E_observed(Shuffled)
```

更强结果：

```text
E_future(FBFM) < E_future(RTC)
E_future(FBFM) < E_future(Shuffled)
```

## 6. 模块 B：验证 Video -> Action

### 6.1 验证问题

FBFM fresh action suffix 的变化是否由 video loop 生成的 predicted KV cache 介导？

### 6.2 使用实验组

- RTC；
- FBFM；
- FBFM-CacheCut；
- RTC+FBFMCache。

### 6.3 Cache 干预边界

每个 action branch 开始前必须恢复完全相同的：

- real-history KV entries；
- cache id/mask 顺序；
- text condition；
- action initial noise；
- action scheduler state；
- action constraint target/mask。

只允许替换 `is_pred=True` 且来源为本 wave video cache-only 调用的 video entries。不能把以下内容一起替换：

- real observation/history entries；
- real action/history entries；
- 其他 wave 的 predicted entries；
- action loop 自己产生的 predicted action entries。

建议在内存中完成 cache snapshot/restore，不把完整多层 KV cache长期写盘。磁盘只保存 cache hash、有效 token 数、`is_pred` 数和少量数值摘要。

### 6.4 Action 比较范围

只比较没有被 hard-overlap action constraint 固定的 fresh 16-action suffix。被约束 prefix 只用于验证三组 action mask 和 target 完全一致。

定义：

```text
D_action(C_F, C_R) = ||A_fresh(C_F) - A_fresh(C_R)||_2
```

报告：

- normalized action L2；
- 反归一化物理动作 L2；
- fresh suffix 最大绝对差；
- 每个 action channel 的 mean/max absolute difference；
- 每个 action solver step 的 velocity difference；
- cache hash、action noise hash 和 action constraint hash。

### 6.5 必要性：CacheCut

定义：

```text
D_cut = ||A_fresh(FBFM-CacheCut) - A_fresh(RTC)||_2
```

如果 action 变化只通过 predicted video cache 传递，则 FBFM-CacheCut 使用 `C_R` 后应退回 RTC action。

### 6.6 充分性：Cache transplant

定义：

```text
D_transplant = ||A_fresh(RTC+FBFMCache) - A_fresh(FBFM)||_2
```

如果 `C_F` 足以传递 action effect，则 RTC+FBFMCache 应复现 FBFM action。

### 6.7 数值一致性基线

在解释 `D_cut` 和 `D_transplant` 前，先运行相同 cache、相同 noise 的重复 action loop，得到 GPU 数值重复误差：

```text
D_repeat = ||A_fresh(repeat1) - A_fresh(repeat2)||_2
```

一致性判据不使用拍脑袋的绝对阈值，而使用预注册重复误差包络：

```text
D_cut <= max(3 * D_repeat, epsilon)
D_transplant <= max(3 * D_repeat, epsilon)
```

`epsilon` 在一次实现 smoke 后根据输出 dtype 固定，正式数据开始后不得调整。

### 6.8 模块 B 的判据

```text
A_fresh(FBFM) != A_fresh(RTC)
A_fresh(FBFM-CacheCut) ~= A_fresh(RTC)
A_fresh(RTC+FBFMCache) ~= A_fresh(FBFM)
```

如果 CacheCut 后 action 仍显著不同，说明还有未隔离的 feedback-to-action 直接路径、cache 污染或 RNG/scheduler 状态不一致，不能进入成功率解释。

## 7. 模块 C：验证 Action -> Success

### 7.1 验证问题

由真实 feedback 和 FBFM video cache引起的 fresh action 变化，是否提高局部任务进展和完整 episode 成功率？

### 7.2 使用实验组

机制子集：

- RTC；
- FBFM；
- FBFM-CacheCut；
- FBFM-Shuffled。

完整主结果：

- RTC；
- FBFM。

RTC+FBFMCache 只用于模块 B 的机制诊断，不作为正式部署方法。

### 7.3 相同 simulator state 的 one-wave 分叉

执行共同的 committed 16-action suffix后，在 fresh suffix 开始前保存完全相同的 simulator branch state。优先保存并恢复：

- robot joint positions/velocities；
- object poses/velocities；
- controller state；
- task-specific internal state；
- simulator RNG state。

如果 RoboTwin task 不支持可靠快照，则采用“相同初始 seed + 重放完全相同历史动作”重建 branch point。重建后必须验证关键 state tensor 和 rendered observation 一致，才能纳入实验。

从相同 branch point 分别执行各组 fresh 16-action suffix。

### 7.4 局部任务指标

主要记录：

- fresh suffix 内是否成功；
- suffix 结束后的 `eval_success`；
- 成功发生在 suffix 内的 action step；
- suffix 后使用统一 RTC continuation policy，在固定额外预算内是否成功；
- suffix 后到成功所需的额外 action steps。

如果 task 暴露稳定的阶段/predicate 状态，可额外报告：

- 已满足 task predicates 数；
- grasp/lift/place/open 等阶段进展；
- task-specific object-to-goal distance。

task-specific dense metric 只能作为辅助指标，不能跨任务直接求无定义的平均值。

### 7.5 完整 rollout 指标

报告：

- 每任务成功数/总 trial 数；
- 每任务 success rate；
- 50-task macro success rate；
- 全 episode micro success rate；
- 平均/中位完成步数；
- timeout rate；
- 成功 episode 的 time-to-success。

### 7.6 模块 C 的判据

```text
Success(FBFM) > Success(RTC)
Success(FBFM) > Success(FBFM-CacheCut)
Success(FBFM) > Success(FBFM-Shuffled)
Success(FBFM-CacheCut) ~= Success(RTC)
```

如果 FBFM 改变了 action 但没有提高局部结果或完整成功率，只能声称“feedback 影响了 action”，不能声称这种影响提高了策略性能。

## 8. 数据规模与任务选择

### 8.1 Smoke 阶段

- 1 个已有成功运行记录的 task；
- 2 个 trial seeds；
- 每个 episode 最多记录前 2 个完整 waves；
- 跑通 RTC、FBFM、Shuffled、CacheCut 和 cache transplant；
- 不用于论文统计。

### 8.2 机制实验阶段

- 8 个预注册 tasks；
- 每个 task 20 个配对 trial seeds；
- 每个 episode 取前 3 个完整且尚未成功的 waves；
- 最多 `8 x 20 x 3 = 480` 个配对 waves。

任务必须在查看辅助实验 FBFM 结果前确定。建议只使用已有 RTC baseline 选择：

1. 排除 RTC 20/20 饱和和 0/20 完全失败任务；
2. 对剩余任务按 RTC success rate、task name 固定排序；
3. 等间隔选择 8 个任务；
4. 把最终任务列表和选择脚本写入 manifest。

如果符合条件的任务少于 8 个，则使用全部符合条件任务，不用 FBFM 结果补选。

### 8.3 完整成功率阶段

- RTC 与 FBFM：沿用完整 50-task、每任务 20 trials 的正式评测；
- FBFM-CacheCut 与 FBFM-Shuffled：至少在相同的 8-task 机制子集上各跑 20 trials；
- 所有方法使用相同 task/trial seed 对。

## 9. 统计分析

### 9.1 配对单位

- latent/action 机制指标：`task + trial_id + wave_id`；
- episode 成功率：`task + trial_id`；
- solver steps 不是独立样本，不能把一个 wave 内的 25 个 numerical steps 当作 25 个独立观测。

### 9.2 连续指标

对 latent error 和 action difference 报告：

- paired mean/median difference；
- task-stratified bootstrap 95% CI；
- 每任务分布；
- 配对散点图，而不只报告总体均值。

### 9.3 二元成功指标

- RTC vs FBFM 使用 paired McNemar test；
- 报告 paired success gain 和 task-stratified bootstrap 95% CI；
- 同时报告 macro 和 micro success rate；
- CacheCut、Shuffled 多重比较使用 Holm correction。

### 9.4 预注册主指标

建议把以下四项设为主指标：

1. slot-0 final normalized latent MSE：FBFM vs RTC；
2. `D_cut / D_repeat` 与 `D_transplant / D_repeat`；
3. one-wave paired success：FBFM vs RTC/CacheCut；
4. 完整 episode paired success：FBFM vs RTC。

slot-1 error、LPIPS、PSNR、task-specific progress 作为辅助指标。

## 10. 需要记录的实验产物

建议目录结构：

```text
results/lingbot_fbfm_causal_aux/
  manifest.json
  waves.jsonl
  episodes.jsonl
  latent_metrics.csv
  action_metrics.csv
  task_summary.csv
  statistical_tests.json
  artifacts/
    task_<name>/trial_<id>/wave_<id>/
      video_rtc.pt
      video_fbfm.pt
      video_shuffled.pt
      action_rtc.npy
      action_fbfm.npy
      action_cachecut.npy
      action_transplant.npy
      feedback_target.pt
      future_target.pt
      cache_summary.json
```

`manifest.json` 至少包含：

- code commit 和 dirty-worktree 状态；
- checkpoint 路径与 hash；
- task list、trial list 和 shuffled mapping seed；
- H/d/s、solver grants 和 feedback cadence；
- video/action noise seed rule；
- state weight、guidance clip 和 CFG；
- GPU、dtype、PyTorch/CUDA 版本；
- primary metrics 和统计检验；
- smoke 后固定的数值重复误差 `epsilon`。

每条 `waves.jsonl` 至少包含：

- task、trial、wave、mode；
- launch action step 和 branch action step；
- feedback arrival solver step；
- state target source trial/wave；
- constraint versions；
- video/action noise hashes；
- real-cache、pred-cache 和 action-mask hashes；
- observed/future latent metrics；
- fresh action metrics；
- one-wave outcome。

## 11. 实现位置建议

默认正式路径必须保持数值不变；所有辅助功能由显式实验开关启用。

### `wan_va/wan_va_server.py`

- 将 video loop 和 action loop 拆成可复用的内部阶段函数；
- 暴露 video final latent、action initial noise 和 loop boundary；
- 增加 paired trace replay；
- 在 action loop 前提供受控的 predicted cache选择；
- 记录 endpoint、constraint version 和 action suffix。

### `wan_va/modules/model.py`

- 增加只针对 `is_pred=True` entries 的 snapshot/restore helper；
- 区分 video predicted entries 和 action predicted entries；
- 对 snapshot 记录 shape、token count 和 hash；
- 不改变 attention、cache allocation 或正式 forward 数学。

### `evaluation/robotwin`

- 增加 committed suffix trace recorder；
- 增加 simulator branch snapshot/replay；
- 实现预注册 shuffled mapping；
- 保存各 action branch 的 one-wave outcome。

### `script/`

- 增加机制实验 runner；
- 增加结果完整性 validator；
- 增加 paired metrics 与统计表聚合脚本。

## 12. 完整性检查

任何正式结果进入统计前，必须通过：

1. RTC/FBFM action targets 和 masks 逐元素一致；
2. paired video branches 的 real-history cache hash 一致；
3. paired video initial noise hash 一致；
4. paired action branches 的 action initial noise hash 一致；
5. feedback arrival solver step 与原始伪异步 trace 一致；
6. RTC 的 state mask 恒为零；
7. FBFM 的 state target 来自当前 wave；
8. Shuffled target 不来自当前 trial/wave；
9. CacheCut action loop 的 video cache hash 等于配对 RTC；
10. Transplant action loop 的 video cache hash 等于配对 FBFM；
11. branch simulator state 在执行 fresh suffix 前一致；
12. 所有 latent/action 有限且无 NaN/Inf。

任何一项失败，该 wave 必须标记为 invalid，不能静默纳入统计。

## 13. 结论强度与允许表述

### 仅模块 A 成立

允许表述：

> Live feedback improves LingBot-VA's next-state prediction under the matched pseudo-asynchronous schedule.

不能表述 feedback 已经提高 action 或成功率。

### 模块 A+B 成立

允许表述：

> The improved video prediction is transmitted to action generation through LingBot-VA's predicted video KV cache.

不能在模块 C 不成立时声称策略性能提高。

### 模块 A+B+C 全部成立

允许完整表述：

> Under a matched deterministic pseudo-asynchronous protocol, real execution feedback reduces LingBot-VA's next-state prediction error. Cache-cut and cache-transplant interventions show that the corrected predicted video cache mediates the resulting change in the fresh action suffix. Counterfactual rollouts from identical simulator states and paired full-episode evaluations show that this action change improves task success.
