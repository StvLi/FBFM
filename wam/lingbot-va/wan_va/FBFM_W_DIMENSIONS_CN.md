# FBFM 中掩码 W 的维度梳理

本文只对照**当前代码**说明 `W` 在不同情况下的维度，不描述理想化联合模型。

相关代码：

- [modeling_rtc_fbfm.py](/c:/Users/zhang/Documents/GitHub/FBFM/fbfm/policies/fbfm/modeling_rtc_fbfm.py)
- [wan_va_server.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/wan_va/wan_va_server.py)

---

## 1. 统一结论

`W` 在代码里对应变量 `weights`，生成位置在：

- `RTCProcessor.denoise_step()`

当前代码里分两大类：

1. **当前实际运行主分支**
   - `prev_chunk_left_over` 是普通 Tensor
   - `weights` 先生成 `(T,)`
   - 再扩展为 `(1, T, 1)`
   - 最终广播到 `(B, T, D)`

2. **代码保留但当前 server 基本不走的联合分支**
   - `prev_chunk_left_over` 是 `RTCPrevChunk`
   - `state_feedback_enabled=True`
   - `weights` 直接构造成 `(1, T, D_joint)`

---

## 2. 情况 A：video loop 当前实际运行

### 触发位置

- [wan_va_server.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/wan_va/wan_va_server.py)
- `_guided_video_step()`

当前传给 `denoise_step()` 的参数是：

```python
prev_chunk_left_over = self.left_over.state
```

所以这里传入的是：

- **普通 Tensor**
- 不是 `RTCPrevChunk`

### `x_t` 的维度

在 `_guided_video_step()` 里：

```python
flat_latents = rearrange(latents, 'b c f h w -> f (c h w)')
```

所以进入 `denoise_step()` 前：

- `x_t.shape = (T, D_video)`

进入 `denoise_step()` 后因为是 2D，会补 batch 维：

- `x_t.shape = (1, T, D_video)`

其中：

\[
D_{video} = C \times H \times W
\]

### `weights` 的生成过程

当前会走普通 Tensor 分支：

```python
weights = (
    self.get_prefix_weights(inference_delay, execution_horizon, chunk_size)
    .to(x_t.device)
    .unsqueeze(0)
    .unsqueeze(-1)
)
```

因此维度变化为：

1. `get_prefix_weights(...)`
   - `(T,)`

2. `unsqueeze(0)`
   - `(1, T)`

3. `unsqueeze(-1)`
   - `(1, T, 1)`

4. 与 `x_t` 相乘时广播到：
   - `(1, T, D_video)`

### 结论

video loop 里当前 `W` 的实际作用维度是：

- 原始存储：`(1, T, 1)`
- 广播后：`(1, T, D_video)`

它的含义是：

- 只沿时间步变化
- 对所有 latent 维度统一广播

---

## 3. 情况 B：action loop 当前实际运行

### 触发位置

- [wan_va_server.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/wan_va/wan_va_server.py)
- `_guided_action_step()`

当前传给 `denoise_step()` 的参数是：

```python
prev_chunk_left_over = self.left_over.action
```

所以这里也是：

- **普通 Tensor**
- 不是 `RTCPrevChunk`

### `x_t` 的维度

在 `_guided_action_step()` 里：

```python
flat_actions = rearrange(actions, 'b c f h w -> f (c h w)')
```

进入 `denoise_step()` 前：

- `x_t.shape = (T, D_action)`

补 batch 后：

- `x_t.shape = (1, T, D_action)`

其中：

\[
D_{action} = action\_dim \times action\_per\_frame \times 1
\]

### `weights` 的生成过程

与 video loop 相同，也走普通 Tensor 分支：

1. 初始：
   - `(T,)`
2. 扩展后：
   - `(1, T, 1)`
3. 广播后：
   - `(1, T, D_action)`

### 结论

action loop 里当前 `W` 的实际作用维度是：

- 原始存储：`(1, T, 1)`
- 广播后：`(1, T, D_action)`

它的含义是：

- 只沿时间步变化
- 对所有 action flatten 维度统一广播

---

## 4. 情况 C：代码中保留的联合 state/action 分支

### 触发条件

只有满足以下条件才会进入：

1. `prev_chunk_left_over` 是 `RTCPrevChunk`
2. `state_feedback_enabled=True`
3. `chunk_state_dim` 已配置
4. `chunk_action_dim` 已配置
5. `x_t.shape[-1] == chunk_state_dim + chunk_action_dim`

### `weights` 的形状

这时不会走 `(T,) -> (1, T, 1)` 的广播逻辑，而是直接创建：

```python
weights = torch.zeros(1, chunk_size, output_dim)
```

所以：

- `weights.shape = (1, T, D_joint)`

其中：

\[
D_{joint} = chunk\_state\_dim + chunk\_action\_dim
\]

### 内部分块

state 段：

```python
weights[:, :t_state_cap, :state_dim] = 1.0
```

action 段：

```python
weights[:, :t_action, state_dim:] = 1.0
```

所以它是：

- 前半部分对应 state 维度
- 后半部分对应 action 维度
- 是文档里最初设想的联合块掩码

### 结论

联合分支里 `W` 的维度是：

- `(1, T, D_joint)`

不是广播来的，而是直接构造出来的三维块掩码。

---

## 5. 情况 D：当前 server 为什么基本不走联合分支

当前 server 里：

- video loop 传的是 `left_over.state`
- action loop 传的是 `left_over.action`

也就是说，两条线分别传的是普通 Tensor，不是联合 `RTCPrevChunk(state+action)` 给同一个 `x_t`。

所以当前实际运行主要是：

- 情况 A
- 情况 B

而不是情况 C。

---

## 6. 一页总结表

| 情况 | 传入 `prev_chunk_left_over` | `x_t` 形状 | `weights` 初始形状 | `weights` 实际作用形状 |
|---|---|---|---|---|
| Video loop 当前实现 | `left_over.state` | `(1, T, D_video)` | `(1, T, 1)` | `(1, T, D_video)` |
| Action loop 当前实现 | `left_over.action` | `(1, T, D_action)` | `(1, T, 1)` | `(1, T, D_action)` |
| 理想联合分支 | `RTCPrevChunk(state+action)` | `(1, T, D_joint)` | `(1, T, D_joint)` | `(1, T, D_joint)` |

---

## 7. 当前最准确的理解

当前代码中，`W` 已经不是单一的联合 `state/action` 掩码，而是：

1. `W_video`
   - 时间前缀掩码
   - 广播到 video latent 维度

2. `W_action`
   - 时间前缀掩码
   - 广播到 action flatten 维度

而最初文档中的联合块掩码 `W_joint` 目前只保留在 `RTCProcessor` 的代码能力中，没有成为当前 Lingbot-VA server 的主运行路径。

