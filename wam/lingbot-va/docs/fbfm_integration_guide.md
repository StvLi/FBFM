# FBFM (Flow-Based Feedback Model) 集成指南

本文档总结了在 WanVA 服务器端集成 FBFM 策略的核心代码改动思路、架构设计以及开发过程中需要特别注意的细节。

## 1. 核心改动思路

### 1.1 调度器增强：`WrapperedFlowMatchScheduler`
为了支持基于反馈的状态修正，原有的 `FlowMatchScheduler` 被封装为 `WrapperedFlowMatchScheduler`。其核心逻辑是在标准的去噪步骤中插入一个**梯度校正环节**。

- **虚时间轴映射**：将 Diffusers 标准的 Sigma 时间轴映射为 RTC (Residual Time Coordinate) 使用的 $\tau$ ($ \tau = 1 - \sigma $)，以匹配 FBFM 的理论公式。
- **动态校正项计算**：
  - 当存在约束状态 (`constrained_y`) 和权重 (`weights`) 时，启用 `torch.enable_grad()`。
  - 计算预测的 $x_1$ ($x_t - \tau \cdot v_t$)。
  - 计算误差 $err = (y_{target} - x_1) \cdot weights$。
  - 通过 `torch.autograd.grad` 反向传播误差至 $x_t$，得到校正方向 `correction`。
- **自适应引导权重**：
  - 引入时间相关的增益系数 $c(\tau)$ 和 $inv\_r2$，在扩散过程早期（$\tau \to 1$）抑制过大的校正，防止发散。
  - 使用 `max_guidance_weight` 进行截断，保证数值稳定性。
- **最终速度场修正**：$v_{result} = v_t - \text{guidance\_weight} \cdot \text{correction}$。

### 1.2 通用去噪函数抽象：`get_denoise_fn`
为了减少代码重复并统一 Video 和 Action 生成的逻辑，提取了 `get_denoise_fn` 高阶函数。

- **模式统一**：通过 `mode` 参数 ('video' / 'action') 自动选择输入字典键 (`latent_res_lst` / `action_res_lst`) 和后处理逻辑。
- **CFG 逻辑内聚**：将 Classifier-Free Guidance 的计算逻辑 ($noise_{cond} + scale \cdot (noise_{cond} - noise_{uncond})$) 统一封装在闭包内，避免在推理循环中重复编写条件判断。
- **可求导接口**：返回的 `denoise_fn` 保持计算图完整，以便外层 Scheduler 进行梯度回传。

### 1.3 反馈机制集成 (`_feedback` & `PrevChunk`)
在 `VA_Server` 中引入了状态反馈队列机制，用于处理多 Chunk 生成时的连续性约束。

- **状态缓存**：通过 `_feedback` 方法接收上一帧的真实观测状态，编码为 Latent 后存入 `self.prev_chunk_left_over` (FBFM 策略对象)。
- **约束注入**：在 `_infer` 循环调用 `scheduler.step` 时，从策略对象中提取 `constrained_y` (目标状态) 和 `weights` (空间/通道权重)，传递给调度器进行实时校正。

---

## 2. 需要额外注意的细节

### 2.1 梯度图管理 (Gradient Graph Management)
- **显式启用梯度**：在 `WrapperedFlowMatchScheduler.step` 中，必须使用 `with torch.enable_grad():` 包裹校正计算块，因为推理模式默认是 `no_grad` 的。
- **retain_graph=False**：在 `torch.autograd.grad` 中设置 `retain_graph=False`，因为每个 timestep 的校正是独立的，无需保留整个扩散过程的图，以节省显存。
- **detach 操作**：输入 `x_t` 在校正前需 `.clone().detach()`，确保梯度只在校正计算块内流动，不污染上游生成历史。

### 2.2 时间轴与 Sigma 边界处理
- **Tau 计算**：注意 $\tau = 1 - \sigma$ 的定义，确保与 FBFM 论文或实现中的符号一致。
- **终态处理**：当 `to_final=True` 或到达最后一步时，$\sigma_{next}$ 应强制设为 0 (或 1，取决于逆过程定义)，此时通常不进行 FBFM 校正，直接输出结果。
- **Padding 对齐**：代码中对 `timesteps` 进行了 `F.pad(..., value=0)` 操作，需确保 `set_timesteps` 生成的序列长度与推理循环严格匹配，防止索引越界。

### 2.3 CFG 与 Batch 维度
- **Batch 重复**：当 `guidance_scale > 1` 时，`_repeat_input_for_cfg` 会将输入 Batch 维度翻倍 (Condition + Uncondition)。
- **切片逻辑**：在 `get_denoise_fn` 中，CFG 计算后需正确切片 (`noise_pred[:1]` 或 `noise_pred[1:]`)，确保输出维度回归到原始 Batch Size，否则会导致后续矩阵运算报错。
- **Cache 一致性**：Transformer 的 KV Cache 创建时需根据 `use_cfg` 动态调整 `batch_size` 参数 (1 或 2)，否则会导致 Cache 命中失败或形状不匹配。

### 2.4 动作空间掩码 (Action Masking)
- **无效通道清零**：在生成循环结束后，必须执行 `actions[:, ~self.action_mask] *= 0`，确保未被训练的动作通道（如固定的基座位置）保持为零，防止模型产生幻觉噪声。
- **归一化一致性**：`preprocess_action` 和 `postprocess_action` 中的分位数归一化 (`q01`, `q99`) 必须与训练时的统计量严格一致，否则校正方向会偏离真实物理空间。

### 2.5 显存优化
- **Offload 策略**：初始化时可根据 `enable_offload` 将 VAE 和 Text Encoder 置于 CPU，仅在推理时临时移至 GPU。
- **Cache 清理**：在每个 Chunk 推理结束或 Reset 时，务必调用 `transformer.clear_cache` 和 `torch.cuda.empty_cache()`，防止长序列推理导致 OOM。

---

## 3. 文件修改清单

| 文件路径 | 修改类型 | 描述 |
| :--- | :--- | :--- |
| `wan_va_server.py` | 新增类 | `WrapperedFlowMatchScheduler`: 实现带梯度校正的调度逻辑 |
| `wan_va_server.py` | 新增方法 | `get_denoise_fn`: 抽象通用的去噪闭包函数 |
| `wan_va_server.py` | 修改方法 | `_infer`: 重构为双循环 (Video/Action)，集成 FBFM 校正调用 |
| `wan_va_server.py` | 新增方法 | `_feedback`: 处理外部状态反馈输入 |
| `wan_va_server.py` | 修改方法 | `infer`: 初始化 `FBFM.PrevChunk` 策略对象 |

## 4. 配置建议

在 `configs` 中建议增加以下参数以控制 FBFM 行为：
- `rtc_config.max_guidance_weight`: 最大校正权重上限 (默认建议 5.0~10.0)。
- `action_guidance_scale`: 动作生成的独立 CFG 系数。
- `constrain_mode`: 约束模式 (如 "Feedback", "Reference")。