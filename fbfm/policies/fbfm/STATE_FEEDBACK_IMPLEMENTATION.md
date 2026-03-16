# RTC 状态反馈扩展：实现说明

本文档描述在 LeRobot RTC 基础上实现的 **VA 状态反馈（State Feedback）** 扩展：在保留「上一 action chunk 剩余约束下一 action chunk」的同时，用**执行上一 chunk 剩余时机器人真实观测到的状态（VAE 编码的 image latent）**约束 VA 模型的 **state chunk** 生成，实现 chunk 内的状态反馈，增强对环境偏差的鲁棒性。

---

## 1. 目标与约定

- **状态定义**：状态指**图像经 VAE 编码器编码后的 latent**，非关节角等本体信息。
- **Train-free**：不修改已有模型参数或结构，仅在推理侧扩展 RTC 的数据流与权重。
- **整体 x**：Guidance 将 state 与 action 视为**同一向量 x**，一次 backward 得到 correction，再对 state/action 段分别施加不同 guidance 上界。
- **适用模型**：需具备与 action 联合 flow matching 的**显式 state 预测**（如 LingbotVA、DreamZero 等），且 chunk 的每步维度为固定的 `chunk_state_dim + chunk_action_dim`。

---

## 2. 配置（`configuration_rtc.py`）

### 2.1 新增 / 与状态反馈相关的字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `state_feedback_enabled` | bool | 是否启用状态反馈，默认 False。 |
| `state_observed_horizon` | int | 状态 prefix 的 horizon（参与权重有效长度上限）。 |
| `state_max_guidance_weight` | float | 状态段 correction 的 guidance 上界。 |
| `chunk_state_dim` | int \| None | **每个 chunk 时间步内 state 的固定维度**，启用 state feedback 时必填。 |
| `chunk_action_dim` | int \| None | **每个 chunk 时间步内 action 的固定维度**，启用 state feedback 时必填；须满足 `chunk_state_dim + chunk_action_dim == x_t.shape[-1]`。 |
| `state_latent_dim` | int \| None | 观测 state latent 的维度（如 VAE 输出），用于校验等，不参与 D 的划分。 |

### 2.2 校验

- 当 `state_feedback_enabled=True` 时：`chunk_state_dim`、`chunk_action_dim`、`state_max_guidance_weight` 均须为正。

---

## 3. 数据结构与入口（`modeling_rtc.py`）

### 3.1 `RTCPrevChunk`

用于携带「上一 chunk 的剩余」及执行时观测到的状态，供 `denoise_step` 使用。

| 属性 | 类型 | 说明 |
|------|------|------|
| `action` | Tensor \| None | 上一 chunk 未执行完的动作，形状 `(T_prev, action_dim)` 或 `(B, T_prev, action_dim)`。 |
| `state` | Tensor \| None | 执行上一 chunk 剩余时观测到的 state latent（VAE 编码），**初始为空**，随执行逐步追加。 |
| `inference_delay` | int | 推理延迟步数。 |
| `execution_horizon` | int \| None | action 的 horizon 覆盖。 |
| `state_observed_horizon` | int \| None | state 的 horizon 覆盖。 |

**方法**

- **`append_state_latent(self, new_latent: Tensor)`**  
  在执行过程中，每得到一帧图像的 VAE 编码即可调用，将 `new_latent` 沿时间维追加到 `state`。支持 1D `(state_dim,)`、2D `(1, state_dim)` 或 `(T_new, state_dim)`。

### 3.2 `prepare_prev_chunk_left_over(...)`

在**执行侧**构造下一次 RTC 所需的 `prev_chunk_left_over`，便于与动作执行对齐。

- **参数**：`action_left_over`、`observed_state_latents`（可选）、`inference_delay`、`execution_horizon`、`state_observed_horizon`。
- **行为**：  
  - 用 `action_left_over` 实例化 `RTCPrevChunk`，**state 初始为 None**。  
  - 若调用方已有一段 `observed_state_latents`，则通过 `append_state_latent` 一次性写入，与「执行中逐帧 append」的布局一致。  
- **返回**：`RTCPrevChunk` 或 `None`（当两者均为 None 时）。

---

## 4. Chunk 内 state / action 排布与 D 维度

- **约定**：每个时间步 `t` 的向量为 **先 state 维，再 action 维**：  
  `x_t[t] = [ state(t): chunk_state_dim 维 | action(t): chunk_action_dim 维 ]`  
  因此 `x_t.shape = (B, T, D)` 且 `D = chunk_state_dim + chunk_action_dim`。
- **D 的划分**：由配置中的 **`chunk_state_dim`** 与 **`chunk_action_dim`** 固定给出，**不随 `state_latent_dim` 变化**。`denoise_step` 内会校验 `chunk_state_dim + chunk_action_dim == x_t.shape[-1]`。

---

## 5. `denoise_step` 扩展逻辑

### 5.1 输入

- `prev_chunk_left_over` 可为 **Tensor**（仅 action，兼容原 RTC）、**RTCPrevChunk**（state + action）或 **None**。
- 当为 `RTCPrevChunk` 且启用 state feedback 且 `state is not None` 时，走「联合 state+action」分支。

### 5.2 联合 prefix（state feedback 时）

- 构建 `combined_prefix`，形状 `(B, T, D)`，每步为 `[state 段; action 段]`：  
  - 用 `RTCPrevChunk.state` 填前 `t_state` 步的 `:chunk_state_dim`；  
  - 用 `RTCPrevChunk.action` 填前 `t_act` 步的 `chunk_state_dim:`；  
  不足处保持为 0。
- 将 `prev_chunk_tensor` 设为 `combined_prefix`，后续与原始 RTC 一致地参与误差与梯度计算。

### 5.3 权重 diag(W)（硬边缘）

- **结构**（与示意图一致）：  
  **[当前已有状态数个 1；补全状态维数用 0；上个 chunk 遗留动作数个 1；补全动作维数用 0]**  
  即按 (t, dim) 的 block-diagonal，每步内：  
  - state 维：`t < t_state_cap` 为 1，否则 0（`t_state_cap` 由已收集的 state 步数与 `state_observed_horizon` 取 min）；  
  - action 维：`t < t_act` 为 1，否则 0。
- **实现**：  
  `weights` 形状 `(1, T, D)`，  
  `weights[:, :t_state_cap, :chunk_state_dim] = 1.0`，  
  `weights[:, :t_act, chunk_state_dim:] = 1.0`，  
  其余为 0。  
  随 `append_state_latent` 的调用，下次 `denoise_step` 时 `t_state` 增大，权重中 state 段的 1 的个数随之增加。

### 5.4 一次 backward 与整体 x

- `err = (prev_chunk_tensor - x1_t) * weights`，`x1_t = x_t - time * v_t`。  
- 对 `x1_t` 做一次 `torch.autograd.grad`，得到与 `x_t` 同形的 `correction`。  
- 不对 state/action 分别做两次 backward，**整体 x 一次梯度**。

### 5.5 Guidance 权重（对称截断）

- **base_guidance_weight**：由 τ 与公式得到，state 与 action 共用同一时间依赖。
- **截断**（对称命名与实现）：  
  - `action_max_guidance_weight`：来自 `max_guidance_weight`；  
  - `state_max_guidance_weight`：来自 `state_max_guidance_weight`；  
  - `action_guidance_weight = min(base_guidance_weight, action_max_guidance_weight)`；  
  - `state_guidance_weight = min(base_guidance_weight, state_max_guidance_weight)`。
- **result**：  
  - state feedback 时：  
    `result = v_t - cat(state_guidance_weight * correction_state, action_guidance_weight * correction_action)`；  
  - 否则：  
    `result = v_t - action_guidance_weight * correction`。

---

## 6. 执行侧使用流程（建议）

1. 在请求下一 chunk 前，用 `prepare_prev_chunk_left_over(action_left_over=..., observed_state_latents=None, ...)` 得到 `RTCPrevChunk`（此时 `state=None`）。
2. 在执行上一 chunk 剩余动作的过程中，每得到一帧图像的 VAE 编码，调用 `rtc_prev_chunk.append_state_latent(latent)`。
3. 将 `rtc_prev_chunk` 作为 `prev_chunk_left_over` 传入策略的 `predict_action_chunk(..., prev_chunk_left_over=rtc_prev_chunk, ...)`；策略内部再传给 `RTCProcessor.denoise_step`。
4. 配置中设置 `state_feedback_enabled=True`，并正确设置 `chunk_state_dim`、`chunk_action_dim`（及可选 `state_observed_horizon`、`state_max_guidance_weight`）。

---

## 7. 文件与依赖

| 文件 | 作用 |
|------|------|
| `configuration_rtc.py` | RTC 与 state feedback 配置项及校验。 |
| `modeling_rtc.py` | `RTCPrevChunk`、`prepare_prev_chunk_left_over`、`RTCProcessor.denoise_step`（含联合 prefix、diag(W)、一次 backward、对称 guidance 截断）。 |
| `action_queue.py` | 动作队列（如 `get_left_over()` 提供 action leftover），与 state 反馈无直接依赖。 |
| `debug_tracker.py` / `debug_visualizer.py` | 调试记录与可视化，可选。 |

策略层（如 LingbotVA / DreamZero）需在 denoise 循环中把 `prev_chunk_left_over`、`inference_delay` 等传入 `denoise_step`，且模型输出联合的 `[state; action]` chunk，并保证 `chunk_state_dim + chunk_action_dim` 与 `x_t.shape[-1]` 一致。
