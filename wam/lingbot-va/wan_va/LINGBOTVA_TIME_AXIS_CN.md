# Lingbot-VA 中 latent frame、action step、τ 和 feedback 的关系

本文只基于当前代码和作者对 issue 的回复做解释。

已确认信息：

- `τ = 4` 指的是 **WAN2.2 VAE 的时间下采样率**

---

## 1. 三个时间尺度

### A. action step

最细粒度的时间单位。

在评测脚本里，每执行一次：

```python
TASK_ENV.take_action(...)
```

就推进一个 action step。

---

### B. latent / video frame

这是 `Lingbot-VA` 在视频侧建模的时间单位。

它不是原始环境每一个动作 step 都对应一帧，而是经过时间下采样后的 frame。

作者回复说明：

\[
\tau = 4
\]

表示：

**1 个 latent/video frame 对应 4 个更细粒度的时间单元。**

---

### C. feedback 采样点

feedback 是 client 在执行 action 过程中回传 observation 的时刻。

它应该服务于：

- `video loop` 的 state feedback

因此它最终要和 latent/video frame 的时间轴对齐，而不是单纯和每个 action step 一一对应。

---

## 2. 当前代码中的对应关系

### server 端

在 [wan_va_server.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/wan_va/wan_va_server.py) 中：

- `frame_chunk_size`
  表示一个 chunk 内有多少个 latent/video frame
- `action_per_frame`
  表示每个 latent/video frame 下有多少个 action step

action 张量的组织方式是：

\[
(B,\ C,\ F,\ H,\ 1)
\]

其中：

- `F = frame_chunk_size`
- `H = action_per_frame`

所以 server 端语义是：

\[
1\ frame \rightarrow action\_per\_frame\ action\ steps
\]

---

### client 端

在 [eval_polict_client_openpi.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py) 中：

```python
assert action.shape[2] % 4 == 0
action_per_frame = action.shape[2] // 4
```

结合作者回复，现在更合理的理解是：

- 这里的 `4` 不是随意硬编码
- 它对应 `τ = 4`
- 这段逻辑是在把 action 的细粒度时间轴折算回 latent/video 对齐尺度

因此，这里的变量名 `action_per_frame` 容易误导。

它更像是在计算一个：

- 与 latent/video 时间轴兼容的 feedback 分组长度

而不是重新定义 server 端的 `action_per_frame`。

---

## 3. 这意味着什么

### A. `frame_st_id`

`frame_st_id` 应理解为：

- latent/video frame 级别的时间偏移

不是：

- action step 级别的偏移

---

### B. `PrevChunk.states`

`PrevChunk.states` 的时间轴应优先和：

- latent/video frame

对齐。

也就是说，state feedback 不是天然按每个 action step 都写一次，而应考虑 `τ = 4` 的时间下采样关系。

---

### C. feedback 频率

feedback 的采样频率不能只凭“每隔多少 action step”来解释，还要满足：

- 最终能够对齐到 latent/video frame 的时间轴

所以当前 client 里那段 `// 4` 更像是一个“时间轴折算”步骤。

---

## 4. 一句话总结

在当前 Lingbot-VA 里：

- `action step` 是最细粒度执行时间
- `latent/video frame` 是经过 `τ = 4` 下采样后的视觉时间单位
- `feedback` 应最终服务于 latent/video 时间轴
- 因此 `frame_st_id`、`PrevChunk.states`、feedback 采样频率都应以 latent/video frame 语义为准，而不是简单按 action step 解释。

