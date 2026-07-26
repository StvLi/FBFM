# DreamZero x FBFM 状态引导问题复盘

更新时间：2026-07-26

本文复盘从发现 DreamZero 状态约束使用 `56/9600`、可能被过度削弱开始，
到当前 RMS-balanced FBFM 在 `libero_object/task_006` 上以 `3/10` 追平
native base 的全过程。本文只汇报已经运行过的代码和实验，并将已证实结论、
工作假设及仍待验证的问题分开。

## 1. 问题与固定实验协议

目标任务为：

```text
LIBERO suite: libero_object
task id: 6
instruction: pick up the butter and place it in the basket
official init ids: 0-9（正式 10-episode 对照）
environment seed: 0
model seed: fixed 0
maximum horizon: 480
hardware: NVIDIA RTX A6000 48 GB
checkpoint: RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000
```

FBFM 使用 `H=16, d=8, s=8` 的伪异步 hard overlap。DreamZero 保留原生
16 个 UniPC scheduler index 和 8 次 DiT evaluation，不增加 DiT 前向次数。
每执行一个环境 action，伪时钟放行一次原生 DiT evaluation。

用于比较的 native base 使用相同任务、init、seed 和 horizon，但采用 DreamZero
原生 `native_sync` rollout。因此该比较回答“完整 FBFM 方法能否恢复到官方 base
水平”，不是只改变 mask 的严格工程消融。后续仍需使用同一
`pseudo_async_overlap` 路径的 `NONE` 模式隔离伪异步 overlap 本身的影响。

## 2. FBFM 在 DreamZero 中的实际计算

令第 `k` 次原生 DiT evaluation 输出未引导速度 `v_k` 和 endpoint Jacobian
`J_k`。对于使用该缓存结果的 UniPC index `j`，当前实现计算：

```text
Xhat_j = X_j - sigma_j * v_k
e_j    = P * W * (Y_j - Xhat_j)
g_j    = J_k^T * e_j
v_j    = v_k - lambda(sigma_j) * g_j
```

其中：

- `W` 表示哪些 state/action 坐标已经被真实观测或执行，支持集仍为二值 hard mask；
- `P` 是跨模态预条件器，动作块系数为 `1`，当前状态块系数为
  `sqrt(56/9600)=0.0763762616`；
- `v_k` 和 `J_k` 只在 DreamZero 原生 8 次 DiT evaluation 时刷新；
- `Xhat_j`、残差、VJP 和引导速度在全部 16 个 UniPC index 上重新计算；
- `prev_predictions` 只缓存未引导的原生 `v_k`，不缓存 `v_j`。

当前代码把状态预条件系数代数等价地乘入 `state_mask`。为避免论文表述混乱，
建议理论部分明确写成“二值 hard-overlap 矩阵 `W` + 模态预条件器 `P`”。如果把
两者合并为一个有效矩阵，则有效状态系数确实为非二值的 `0.076376`。

## 3. 为什么最初的 `56/9600` 有问题

一轮 overlap 中被约束的物理 action 坐标数为：

```text
N_action = 8 actions * 7 coordinates = 56
```

一个 DreamZero state latent slot 的坐标数为：

```text
N_state = 48 channels * 10 * 20 = 9600
```

旧系数：

```text
c_L1 = 56 / 9600 = 0.00583333
```

它使两个 mask 块的元素总和相等，即进行 L1 mass equalization。但 FBFM 的 VJP
在欧氏空间中传播。若暂时假设各坐标残差独立同方差且 Jacobian 为单位阵，则
两个块的期望平方范数分别与 `N_action` 和 `N_state*c^2` 成正比。欧氏能量平衡应为：

```text
N_state * c_RMS^2 = N_action
c_RMS = sqrt(56 / 9600) = 0.0763762616
```

因此 `56/9600` 相对于 action 块会导致：

```text
state/action squared-energy ratio = 56/9600 = 1/171.4
state/action norm ratio           = sqrt(56/9600) = 1/13.1
```

它确实把状态反馈压得很弱。但直接将状态系数改为二值 `1.0` 又走向另一个极端：
在进入真实 Jacobian 前，state 块的期望范数约为 action 块的 `13.1` 倍、平方能量
约为 `171.4` 倍。真实视频 latent、action 的数值分布和 `J_k` 的奇异值还会进一步
改变比例，所以“mask 是二值”本身并不能保证跨模态 VJP 尺度合理。

## 4. 尝试一：恢复二值状态 mask

提交：

```text
f71ec2e  fix(dreamzero): restore binary state guidance mask
e04459b  A6000 deployment of the same method
```

目的：去掉 `56/9600`，令 state/action overlap 均使用系数 `1`，检验旧实现是否因
状态约束过弱而无法发挥作用。

结果：

```text
binary-state FBFM: 0/20 success
mean episode steps: 480
mean episode time: 121.16 s
```

第一段 8 个 action 与旧版本完全一致，轨迹从第一段包含 feedback 的后缀开始分叉，
排除了 checkpoint 加载、reset、初始观测和初始噪声不一致。分叉后动作快速超出
正常分布，例如 trial 0 的两个后续 chunk action norm 达到 `11.15` 和 `22.34`，
而旧弱状态版本对应为 `3.30` 和 `3.19`。

结论：`56/9600` 的确过弱，但二值系数 `1` 并不是正确修复；它暴露并放大了此前
被弱系数掩盖的时序目标和 solver 缓存问题。

## 5. 尝试二：分解 VJP，定位动作爆炸来源

提交：

```text
78f37cc / fe7bae5  decompose joint FBFM VJP
```

诊断把 joint VJP 分解为 state-to-action、state-to-state、action-to-action 和
action-to-state 四个块，不改变默认推理路径。

二值 hold-forward 版本单 episode 的关键结果：

| 指标 | 结果 |
| --- | ---: |
| state-to-action correction mean | 19.94 |
| action-to-action correction mean | 3.06 |
| state-to-action correction max | 2665.07 |
| guided action velocity max | 26641.54 |
| episode | 0/1，480 步失败 |

最强异常集中在早期高噪声 solver step。旧 rolling encoder 会把
`[0,2,3,3,3]` 这类窗口编码为 hard target，其中重复的末帧实际是尚未观测未来
位置的 hold-forward 填充。状态残差通过 DreamZero joint DiT 的 cross-modal
Jacobian 传播到 action 块，成为动作爆炸的主要来源。

结论：问题不在 action hard overlap 自身，而在“不正确或过早的 state target”经
`J_state->action` 被大幅放大。必须保证 state feedback target 是因果、已观测且与
checkpoint 的训练采样方式对齐。

## 6. 尝试三：因果 rolling-past 编码

提交：

```text
e1461c2  use causal rolling-past feedback
```

改动：不再把最新观测复制到未观测未来位置；不足的历史从真实 anchor 向左填充。
第一批窗口从不因果的 hold-forward 改为类似 `[0,0,0,0,1]` 的 causal past。

诊断改善：

| 指标 | hold-forward | causal rolling-past |
| --- | ---: | ---: |
| state-to-action mean | 19.94 | 16.86 |
| state-to-action max | 2665.07 | 424.54 |
| guided action velocity max | 26641.54 | 4243.08 |

但非诊断成功率测试仍为 `0/9`，随后停止。原因是该版本仍按 action stride 1/2
构造视频窗口，与 DreamZero LIBERO SFT checkpoint 的视频采样 stride 不一致。

结论：修正因果性显著降低极值，但只修正“看见了什么”还不够，还必须对齐“训练时
隔多少 action 看一帧”。

## 7. 尝试四：对齐 checkpoint 的 stride-3 视频采样

提交：

```text
be61258  align feedback VAE to training stride
```

训练代码的视频 micro-frame offset 为 `(0,3,6,9,12,15,18,21)`。运行时因此保留
每一步真实 observation，但只在 action offset 3、6、9、12 刷新 VAE hard target。
在一轮 8-action overlap 内，状态窗口依次为：

```text
after action 1-2: state mask remains zero
after action 3:   [0,0,0,0,3]
after action 6:   [0,0,0,3,6]
```

相比 causal stride-2 版本：

| 指标 | causal stride-2 | training-aligned stride-3 |
| --- | ---: | ---: |
| state-to-action mean | 16.86 | 13.30 |
| state-to-action max | 424.54 | 148.53 |
| guided action velocity max | 4243.08 | 1481.42 |

这避免在 `sigma=0.999` 和 `0.986` 的最早两个高噪声 evaluation 激活状态约束。
但执行动作仍然恶化：physical action norm mean/max 为 `2.595/20.214`，最大单坐标
绝对值为 `17.906`。因此没有直接开始 10 次成功率测试。

结论：时序对齐进一步降低单次 VJP 极值，但错误动作一旦进入 8-action hard overlap，
会在后续 wave 中作为已提交条件继续传播。仍存在独立的 solver cache 集成问题。

## 8. 尝试五：修复 DreamZero DiT/UniPC 缓存边界

DreamZero 做 16 次 UniPC scheduler update，但只做 8 次原生 DiT evaluation。
旧 FBFM hook 将 guided velocity 写入 `prev_predictions`，导致一次状态修正在跳过
DiT 的 scheduler index 被重复使用。例如 evaluation 2 会服务 index 2-5，
evaluation 3 会服务 index 6-9；stride-3 的首次状态约束恰好在 evaluation 2 激活。

提交：

```text
9d69cbc  recompute guidance at cached UniPC steps
```

改动：

1. `prev_predictions` 只缓存原生未引导 DiT velocity；
2. guided velocity 只用于当前 UniPC update；
3. index 3/4/5 等跳过 DiT 的位置，使用当前 `X_j` 和 `sigma_j` 重新计算
   `Y-Xhat_j` 与 VJP；
4. 这些位置复用最近一次 DiT endpoint Jacobian，不增加 DiT 计算；
5. 到 index 6 等原生 DiT evaluation 时同时刷新 velocity 和 Jacobian。

这完成了预期的 solver 语义，但在状态系数仍为 `1.0` 时，二值版本出现更明显的
闭环正反馈：

| 指标 | binary relinearized result |
| --- | ---: |
| state-to-action correction mean | 104.22 |
| state-to-action correction max | 19449.25 |
| guided action velocity max | 195700.41 |
| physical action norm mean/max | 11.453 / 391.609 |
| episode | 0/1，480 步失败 |

这不是缓存修复无效，而是修复后每个 scheduler index 都忠实重算了一个本来就尺度
过大的状态修正，使正反馈不再被旧缓存路径偶然稀释。

另一个候选提交 `fa4c2ca` 尝试用前一 index 的 guided velocity 递归构造下一次
endpoint。其 effective action velocity 很快超过 `1.08e6`，证明 guided velocity
不能进入 endpoint 基准或 DreamZero 原生缓存。该尝试已由 `218239f` 明确回滚。

## 9. 尝试六：采用 RMS 坐标平衡状态系数

提交：

```text
13de791  RMS-balance state guidance
fb701b1  update validation handover
```

保留 `9d69cbc` 的 relinearized UniPC/Jacobian 行为，将状态系数从 `1.0` 改为：

```text
sqrt(56/9600) = 0.07637626158259733
```

### 9.1 单 episode VJP 数值门控

| 指标 | binary `1.0` | RMS `0.076376` |
| --- | ---: | ---: |
| state-to-action correction mean | 104.22 | 1.377 |
| state-to-action correction max | 19449.25 | 62.349 |
| guided action velocity max | 195700.41 | 627.031 |
| physical action norm mean | 11.453 | 1.224 |
| physical action norm max | 391.609 | 1.485 |
| outcome | 0/1 failure | 1/1 success，301 步 |

首次激活状态约束的 cached block（index 2-5）中，state residual norm 稳定在
`6.74-7.01`，对应 state-to-action correction 为 `0.581-0.592`，没有再发生
逐 index 爆炸。

### 9.2 正式 10-episode 对照

| 方法 | 成功 | 成功 trial | 平均步数 | 平均 episode 时间 | 平均 wave 时间 |
| --- | ---: | --- | ---: | ---: | ---: |
| native DreamZero base | 3/10 | 0, 6, 8 | 414.8 | 62.21 s | 1.122 s |
| RMS-balanced FBFM | 3/10 | 2, 6, 7 | 399.8 | 123.16 s | 2.389 s |

两者成功率点估计均为 `30%`，相同的 95% Wilson 区间为 `10.8%-60.3%`。配对结果中
只有 trial 6 共同成功；FBFM-only 为 trial 2、7，base-only 为 trial 0、8。
因此 FBFM 改变了闭环轨迹，而不是数值上退化成 base。

正式 FBFM 运行的稳定性：

```text
3998 executed actions
action norm mean / P95 / max: 1.171 / 1.383 / 8.693
maximum absolute action coordinate: 7.969
server errors: 0
GPU allocated first/last 200 evaluations: 26.003 / 26.081 GiB
GPU allocated peak: 27.250 GiB
```

显存没有随推理波次线性增长，说明当前 feedback VAE、VJP 和缓存路径没有保留计算图。
仍有少量 action outlier，需要在更大样本中继续观察。FBFM 平均 wall-clock 时间约为
native base 的 `1.98x`；这是工程开销指标，不参与伪异步数学时钟定义。

## 10. 当前可以得出的原理性结论

### 10.1 FBFM 的约束强度不能只由 mask 是否二值决定

二值 `W` 只定义 hard-overlap 的支持集。视频 latent 和 action 的维度、尺度以及
cross-modal Jacobian 完全不同。如果不做模态预条件，9600 维状态块会在欧氏 VJP
中压倒 56 维 action 块。`P_state=sqrt(56/9600)` 是在简单等方差/单位 Jacobian
假设下的第一阶坐标平衡，不是经验性把反馈任意调小。

### 10.2 L1 mask mass 与 VJP 的 L2 能量不是同一件事

旧 `56/9600` 只平衡 mask 元素之和，在欧氏修正范数下使状态比 action 弱约 13 倍。
`1.0` 又使状态在进入 Jacobian 前强约 13 倍。RMS 系数正好位于两者之间，并在本次
数值诊断中消除了正反馈爆炸。

### 10.3 state target 的语义比“每步都反馈”更重要

每次执行 action 后都可以接收 observation，但不能把未观测未来帧伪装为 hard
target。rolling feedback 必须同时满足：因果、仅使用已观测帧、与 checkpoint
训练视频 stride 对齐。当前实现保留每步 observation，在 stride-3 位置刷新 latent，
这并不等价于丢弃中间环境状态。

### 10.4 solver 加速缓存必须缓存原生场，不能缓存受约束后的场

DreamZero 复用 DiT velocity 是模型自身的加速设计。FBFM guidance 是当前 solver
状态和当前约束的函数，不能作为新的原生 velocity 写回跨 index 缓存。正确做法是
缓存 native `v_k/J_k`，每个 index 重算 endpoint residual 和 VJP；否则同一高噪声
修正会被重复积分或递归放大。

### 10.5 hard action overlap 会传播错误，也会提供必要一致性

hard overlap 本身不是本轮发现的首个故障源，但它会把异常生成的 committed action
带入下一轮约束，形成闭环正反馈。因此必须先通过 VJP/action-norm 数值门控，再进行
昂贵的成功率测试。只看单次 correction 平均值不足以判断安全性，必须同时检查极值、
最终物理 action 以及跨 wave 演化。

## 11. 尚不能得出的结论与后续实验

当前 `3/10` 只能说明：在这个任务和这 10 个官方 init 上，RMS-balanced 版本已从
binary-state 的 `0/20` 灾难恢复到 native base 的相同点估计。它不能证明：

- FBFM 已在整个 LIBERO benchmark 上追平或超过 base；
- 两种方法真实成功率相等；10 次的 Wilson 区间仍很宽；
- 当前 `0.076376` 是最优系数；它只是有理论依据且通过数值门控的默认点；
- 先前队友报告的约 `90%` 与这里属于同一任务、checkpoint、seed 和 rollout 协议。

正在把同一任务扩展到官方 init 0-19。完成后应至少报告以下三组 20-episode 结果：

1. native base：官方 DreamZero rollout；
2. matched `NONE`：同一 pseudo-async overlap 工程路径但零 guidance；
3. RMS FBFM：action hard overlap + stride-3 rolling state feedback。

可再增加 `RTC` 和状态系数 `{56/9600, sqrt(56/9600), 1.0}` 消融。这样才能把
native rollout 差异、action overlap、state feedback 和状态预条件器的影响分开。

## 12. 代码与数据索引

```text
active branch:
  fix/dreamzero-relinearized-unipc-guidance

active code commit:
  13de791

active branch tip / handover commit:
  fb701b1

A6000 repository:
  /home/deepcybo-lite/peize/DreamZero-FBFM-LIBERO

RMS FBFM result:
  results/libero_object6_rms00764_fbfm_10_13de791

native base result:
  results/libero_object6_matched_base_10_e04459b

detailed experiment ledger:
  /home/oem/tmp_ws/aaai_paper/experiments/
  dreamzero_object6_binary_mask_diagnosis.md
```
