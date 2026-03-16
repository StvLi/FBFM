# `RTCProcessor.denoise_step(...)` 伪代码说明

本文只描述当前代码中 `RTCProcessor.denoise_step(...)` 的逻辑，不描述外部 server/client 调用流程。

相关代码：

- [modeling_rtc_fbfm.py](/c:/Users/zhang/Documents/GitHub/FBFM/fbfm/policies/fbfm/modeling_rtc_fbfm.py)

---

## 1. 先说结论

`denoise_step(...)` 是一个**统一的前缀修正器**：

- 给它当前流变量 `x_t`
- 给它一个前缀 `prev_chunk_left_over`
- 给它原始去噪器 `original_denoise_step_partial`

它就会返回一个**带引导的 velocity**

```text
result = 原始 velocity + / - 前缀约束修正项
```

这个函数本身不直接知道“这是 video loop 还是 action loop”，它只根据：

- `prev_chunk_left_over` 的类型
- `x_t` 的维度
- 配置项

在内部进行分支判断。

---

## 2. 输入

函数签名可概括为：

```python
denoise_step(
    x_t,
    prev_chunk_left_over,
    inference_delay,
    time,
    original_denoise_step_partial,
    execution_horizon=None,
)
```

输入意义：

- `x_t`
  当前要更新的 flow 变量
- `prev_chunk_left_over`
  前缀信息，可能是：
  - `Tensor`
  - `RTCPrevChunk`
  - `None`
- `inference_delay`
  前缀起始时间位置
- `time`
  当前 flow 时间
- `original_denoise_step_partial`
  原始 velocity 计算函数
- `execution_horizon`
  RTC horizon

---

## 3. 伪代码总览

```text
function denoise_step(x_t, prev_chunk_left_over, inference_delay, time, original_denoise, execution_horizon):

    tau = 1 - time

    # Step A. 解析 prev_chunk_left_over 类型
    if prev_chunk_left_over is RTCPrevChunk:
        读取 prev_action
        读取 prev_state
        prev_chunk_tensor = prev_action
        如果 RTCPrevChunk 里有 execution_horizon:
            用它覆盖输入 execution_horizon
        inference_delay = prev_chunk_left_over.inference_delay
    else:
        if prev_chunk_left_over is None:
            return original_denoise(x_t)
        prev_chunk_tensor = prev_chunk_left_over

    # Step B. 统一维度到三维
    x_t = clone/detach(x_t)
    if x_t 是 2D:
        x_t -> (1, T, D)
    if prev_chunk_tensor 存在且是 2D:
        prev_chunk_tensor -> (1, T, D_prefix)

    # Step C. 确定 execution_horizon
    if execution_horizon is None:
        execution_horizon = config.execution_horizon
    if prev_chunk_tensor exists:
        execution_horizon = min(execution_horizon, prev_chunk_tensor.shape[1])

    读取:
        batch_size = x_t.shape[0]
        chunk_size = x_t.shape[1]
        output_dim = x_t.shape[2]

    # Step D. 如果传入 RTCPrevChunk，尝试判断该用 state 还是 action
    if prev_chunk_left_over is RTCPrevChunk:
        检查 prev_state 与 output_dim 是否兼容
        检查 prev_action 与 output_dim 是否兼容

        if state 与当前 x_t 精确匹配，action 不精确匹配:
            prev_chunk_tensor = prev_state
        elif 当前没有 prev_chunk_tensor 且 state 可兼容:
            prev_chunk_tensor = prev_state
        elif 当前 prev_chunk_tensor 不兼容而 state 可兼容:
            prev_chunk_tensor = prev_state

    # Step E. 判断是否进入联合 state/action 分支
    use_state_feedback =
        prev_chunk_left_over 是 RTCPrevChunk
        and config.state_feedback_enabled
        and config.chunk_state_dim 不为空
        and config.chunk_action_dim 不为空
        and RTCPrevChunk.state 不为空

    if prev_chunk_tensor is None and not use_state_feedback:
        return original_denoise(x_t)

    # Step F. 构造 prefix 与 W
    if use_state_feedback:
        构造 combined_prefix of shape (B, T, D)
        填 state 段
        填 action 段
        prev_chunk_tensor = combined_prefix

        构造联合三维 W:
            weights.shape = (1, T, D)
            state 段按 t_state_cap 置 1
            action 段按 t_action 置 1
    else:
        构造普通 RTC 时间前缀权重:
            weights.shape = (1, T, 1)

    # Step G. 如有必要，对 prev_chunk_tensor 做右侧补零
    如果 prev_chunk_tensor 的 T 或 D 小于 x_t:
        pad 到与 x_t 相同

    # Step H. 计算原始 velocity 与 correction
    v_t = original_denoise(x_t)
    x1_t = x_t - time * v_t
    err = (prev_chunk_tensor - x1_t) * weights
    correction = grad(x1_t w.r.t x_t, grad_outputs=err)

    # Step I. 计算 guidance 权重
    base_guidance_weight = tau 相关函数
    action_guidance_weight = min(base_guidance_weight, config.max_guidance_weight)
    state_guidance_weight  = min(base_guidance_weight, config.state_max_guidance_weight)

    # Step J. 输出结果
    if use_state_feedback:
        result = v_t - [state_guidance_weight * correction_state,
                        action_guidance_weight * correction_action]
    else:
        result = v_t - action_guidance_weight * correction

    # Step K. 去掉临时加的 batch 维
    if 输入原本是 2D:
        squeeze 回去

    记录 debug
    return result
```

---

## 4. 关键分支说明

## 分支 1：`prev_chunk_left_over` 是 `None`

伪代码：

```text
if prev_chunk_left_over is None:
    return original_denoise(x_t)
```

含义：

- 没有任何 prefix
- 直接返回原始 velocity
- 不做 RTC
- 不做 feedback 修正

---

## 分支 2：`prev_chunk_left_over` 是普通 Tensor

伪代码：

```text
prev_chunk_tensor = prev_chunk_left_over
```

后续：

- 不走联合 state/action 分支
- 构造 `weights = (1, T, 1)`
- 按普通 RTC 时间前缀权重处理

当前 Lingbot-VA 主运行就是这条：

- video loop 传 `left_over.state`
- action loop 传 `left_over.action`

所以在当前运行里：

- “state feedback” 和 “action RTC” 的区别，不是 `denoise_step()` 内部显式判断出来的
- 而是调用方传进来不同 Tensor 决定的

---

## 分支 3：`prev_chunk_left_over` 是 `RTCPrevChunk`

伪代码：

```text
if isinstance(prev_chunk_left_over, RTCPrevChunk):
    prev_action = prev_chunk_left_over.action
    prev_state = prev_chunk_left_over.state
    prev_chunk_tensor = prev_action
```

然后函数会进一步判断：

### 3.1 先看当前 `x_t` 和谁更兼容

```text
if prev_state 的维度与 output_dim 更匹配:
    prev_chunk_tensor = prev_state
elif prev_action 更匹配:
    prev_chunk_tensor 保持为 prev_action
```

也就是说：

- 传整个 `RTCPrevChunk` 进来
- 不代表一定走联合分支
- 也可能退化成只用 state
- 也可能退化成只用 action

### 3.2 再看是否满足联合分支条件

```text
use_state_feedback =
    isinstance(prev_chunk_left_over, RTCPrevChunk)
    and state_feedback_enabled
    and chunk_state_dim is not None
    and chunk_action_dim is not None
    and prev_chunk_left_over.state is not None
```

如果 `use_state_feedback=False`：

- 仍然走普通 Tensor RTC 路径

如果 `use_state_feedback=True`：

- 进入联合 state/action 分支

---

## 5. 普通 Tensor 路径的伪代码

```text
weights = get_prefix_weights(...).unsqueeze(0).unsqueeze(-1)
weights.shape = (1, T, 1)

v_t = original_denoise(x_t)
x1_t = x_t - time * v_t
err = (prev_chunk_tensor - x1_t) * weights
correction = autograd.grad(...)

result = v_t - action_guidance_weight * correction
```

特点：

- `weights` 只随时间变化
- 对所有特征维统一广播
- 当前 Lingbot-VA 的 video/action 两条主路径都走这个逻辑

---

## 6. 联合 state/action 路径的伪代码

```text
构造 combined_prefix of shape (1, T, D_joint)

combined_prefix[:, :t_state, :state_dim] = state_prefix
combined_prefix[:, :t_action, state_dim:state_dim+d_action] = action_prefix

weights = zeros(1, T, D_joint)
weights[:, :t_state_cap, :state_dim] = 1
weights[:, :t_action, state_dim:] = 1

v_t = original_denoise(x_t)
x1_t = x_t - time * v_t
err = (combined_prefix - x1_t) * weights
correction = autograd.grad(...)

result =
    v_t
    - concat(
        state_guidance_weight * correction_state,
        action_guidance_weight * correction_action
      )
```

特点：

- `weights` 是真正的三维块掩码
- state 和 action 段分别掩码
- 这是你最初文档想表达的那套逻辑

---

## 7. 当前代码里“RTC”和“feedback”到底是谁决定的

这点很重要。

`denoise_step(...)` 本身不会写：

```text
if this is action: 做RTC
if this is state: 做feedback
```

它内部没有这种语义判断。

当前代码中：

- **RTC / feedback 的语义由调用方决定**
- `denoise_step(...)` 只负责做“prefix guidance 数学计算”

也就是：

### 外部调用方决定语义

```text
_guided_video_step():
    x_t = 当前 video latent
    prev_chunk_left_over = left_over.state
    => 这次调用被解释为“state feedback”

_guided_action_step():
    x_t = 当前 action chunk
    prev_chunk_left_over = left_over.action
    => 这次调用被解释为“RTC action guidance”
```

### 函数内部只做统一修正

```text
给定 x_t + prefix + 原始 denoise
=> 算 correction
=> 输出 guided velocity
```

---

## 8. 当前 Lingbot-VA 实际会走哪条伪代码

当前 server 实际主要走：

### video loop

```text
调用方传 left_over.state (Tensor)
=> denoise_step 走普通 Tensor 路径
=> 被外部语义解释为“state feedback guidance”
```

### action loop

```text
调用方传 left_over.action (Tensor)
=> denoise_step 走普通 Tensor 路径
=> 被外部语义解释为“RTC action guidance”
```

### 当前基本不走

```text
调用方传整个 RTCPrevChunk
并满足 state_feedback_enabled + 联合维度配置
=> 走联合 state/action 分支
```

---

## 9. 最简图式总结

```text
调用方
  ├─ 传 left_over.state  ──> denoise_step()
  │                         └─ 普通 Tensor 路径
  │                         └─ 在外部语义上表示 state feedback
  │
  ├─ 传 left_over.action ──> denoise_step()
  │                         └─ 普通 Tensor 路径
  │                         └─ 在外部语义上表示 RTC action guidance
  │
  └─ 传 RTCPrevChunk 整体 ─> denoise_step()
                            ├─ 可能退化成 state-only
                            ├─ 可能退化成 action-only
                            └─ 满足条件时走联合分支
```

