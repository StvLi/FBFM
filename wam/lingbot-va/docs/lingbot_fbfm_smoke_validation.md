# LingBot-VA + FBFM RoboTwin 链路验证记录

日期：2026-07-24

## 目标与判据

本轮只验证 FBFM 数学链路，不用 wall-clock 延时决定任何方法变量。三个模式使用同一
checkpoint、RoboTwin seed、动作执行和 solver 预算：

这里的 `NONE` 是同一重叠伪异步 rollout 内的零约束基线，不是 LingBot 官方一次执行完整新 chunk
的同步评测。它用于隔离 FBFM/RTC mask 的数学作用；checkpoint、state-first/action-second solver、
VAE、KV 和 prediction-cache 语义仍与上游一致。

| 模式 | state constraint | previous-action constraint |
|---|---|---|
| NONE | 0 | 0 |
| RTC | 0 | RTC mask |
| FBFM | 动态真实 latent slot | 与 RTC 完全相同 |

链路通过必须同时满足：checkpoint/server 能启动；输出 action/latent 有限；真实
observation/action 只写入 KV 一次；NONE 保持上游数值路径；RTC 的 action correction 非零；FBFM
在反馈到达后的 active video solver step 上出现非零 state mask、error 和 VJP correction；chunk
间显存不线性增长。

## 固定实验坐标

- task：`adjust_bottle`，配置 `demo_clean`，seed `0`
- checkpoint：`/home/oem/tmp_ws/checkpoints/lingbot-va-posttrain-robotwin`
- action horizon：`H=32`，固定 hard boundary `d=16`，execution horizon `s=16`
- video flow：25 个数值步 + 1 个 cache-only evaluation
- action flow：50 个数值步 + 1 个 cache-only evaluation
- 伪异步时钟：16 个 simulation steps 固定释放 26 个 video evaluations
- feedback encoding：每 4 个 observation 形成 1 个 video latent slot
- video CFG：5；action CFG：1；checkpoint 与全部模型参数冻结

默认离散时序下，前 15 个 simulation steps 释放 24 个 video evaluations；第 16 步形成第一个
完整 feedback latent 后释放最后 2 个 evaluations。因此真实状态反馈作用于最后 1 个数值步，
随后 cache-only evaluation 把修正后的 latent 写入 LingBot 原生 prediction cache。这个位置由固定
伪时钟决定，不由主机速度测量得到。

## 启动前验证

| 项目 | 结果 |
|---|---|
| checkpoint 结构和张量 | PASS：23 GiB；transformer/text encoder/VAE/tokenizer 均可构造 |
| LingBot 环境 | PASS：Python 3.10.16，PyTorch 2.9.0+cu129，CUDA 可用 |
| RoboTwin 环境 imports | PASS：SAPIEN 3.0.0b1，MPLib 0.2.1，Gymnasium 0.29.1 |
| CPU 数学与协议回归 | PASS：`39 passed` |
| NONE/RTC/FBFM mask replay | PASS |
| RoboTwin assets | PASS：11,000 个背景纹理、9,368 个 object 文件、229 个 embodiment 文件；CuRobo 路径已更新 |

## GPU smoke 结果

GPU smoke 前暂停可恢复的 DreamZero x LIBERO benchmark；三组结束后再恢复。

| 模式 | episode | success | state VJP | action VJP | finite | peak GPU | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| NONE | 1 | 0/1 | 0 | 0 | PASS | 17,236.0 MiB allocated | 链路和零约束退化 PASS；单样例失败不作成功率估计 |
| RTC | 1 | 1/1 | 0 | 非零 | PASS | 18,942.1 MiB allocated | state/action 隔离 PASS |
| FBFM | 1 | 1/1 | 8 个 active-chunk 记录非零 | 非零 | PASS | 21,342.1 MiB allocated | 动态反馈、VJP 和 checkpoint 闭环 PASS |

对应输出：

- NONE：`robotwin_outputs/adjust_bottle_NONE_20260724_101428/`
- RTC：`robotwin_outputs/adjust_bottle_RTC_20260724_101754/`
- FBFM：`robotwin_outputs/fbfm_20_20260724_102818/shard0/` 的 seed 10000

三行用于链路验收，不是成功率对比。FBFM 行来自 20-trial 分片运行，其 instruction candidate
数量与单次 NONE/RTC smoke 不同；正式消融必须让三个模式使用相同 trial count、seed block 和 prompt
生成协议。

服务端日志中的关键审计行以 `FBFM solver diagnostics` 和 `FBFM GPU memory` 开头。
