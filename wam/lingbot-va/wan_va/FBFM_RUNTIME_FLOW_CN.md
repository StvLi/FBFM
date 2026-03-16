# FBFM 当前实现时序图式说明

本文只描述**当前代码实际实现**，不描述理想化的联合 `[state; action]` 单变量流匹配。

相关代码：

- [wan_va_server.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/wan_va/wan_va_server.py)
- [eval_polict_client_openpi.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py)

---

## 1. 整体时序

```text
RoboTwin环境/客户端                                Lingbot-VA服务端
────────────────────────────────────────────────────────────────────────────
1. reset=True, prompt ──────────────────────────> infer()
                                                  └─ _reset()
                                                     清空:
                                                     - frame_st_id
                                                     - init_latent
                                                     - left_over
                                                     - last_generated_action_chunk

2. obs=first_obs ───────────────────────────────> infer()
                                                  ├─ fbfm_enabled=False:
                                                  │   └─ _infer()
                                                  └─ fbfm_enabled=True:
                                                      └─ _infer_guided()
                                                         ├─ video loop
                                                         └─ action loop
                                                  <──── 返回 action chunk

3. 客户端执行 action chunk

4. 每到 key frame:
   TASK_ENV.get_obs()
   → format_obs(obs)
   → feedback=True ─────────────────────────────> infer()
                                                  └─ _feedback()
                                                     └─ 编码真实观测
                                                     └─ append 到 left_over.state

5. chunk 执行完:
   obs=key_frame_list, prev_action=action,
   compute_kv_cache=True ───────────────────────> infer()
                                                  └─ _compute_kv_cache()
                                                     ├─ 编码 obs -> latent
                                                     ├─ preprocess prev_action
                                                     └─ 更新 transformer cache

6. 下一次 obs=... ─────────────────────────────> infer()
                                                  └─ 再次生成下一 chunk
                                                     这时:
                                                     - video loop 用 left_over.state
                                                     - action loop 用 left_over.action
```

---

## 2. 服务端内部持久状态

服务端跨请求维护以下变量：

- `self.left_over`
  - `left_over.state`：执行当前 chunk 时收到的真实观测反馈，经 VAE 编码后的 latent
  - `left_over.action`：上一次生成出的 action chunk，flatten 后保存下来，供下一次 RTC action guidance 使用
- `self.last_generated_action_chunk`
  - 最近一次生成完成、但尚未 `postprocess` 前的 action tensor
- `self.frame_st_id`
  - 当前 cache / grid id 的时间偏移
- `self.init_latent`
  - 初始观测编码得到的 latent

---

## 3. 两条修正线路

### 线路 A：video loop 中的状态反馈修正

```text
真实观测 obs
  └─ 客户端 format_obs()
      └─ feedback=True 发给 server
          └─ server._feedback()
              └─ _encode_obs(obs)
                  └─ 得到 video latent
                      └─ append 到 left_over.state

下一次 _infer_guided():
  └─ _guided_video_step()
      ├─ 读取 left_over.state
      ├─ 若没有 state feedback:
      │   └─ 退化为普通 video latent 更新
      └─ 若有 state feedback:
          ├─ 当前 latents flatten 成 (F, D)
          ├─ 调 RTCProcessor.denoise_step(...)
          └─ 修正 video latent 的 flow step
```

这里修正的是：

- **未来 video latent / state chunk**

这里使用的数据是：

- `left_over.state`

这里**不直接用**：

- `left_over.action`

---

### 线路 B：action loop 中的 RTC leftover 修正

```text
当前一次 infer() 结束后:
  └─ server 把 last_generated_action_chunk
      flatten 后写入 left_over.action

下一次 _infer_guided():
  └─ _guided_action_step()
      ├─ 读取 left_over.action
      ├─ 若没有 action leftover:
      │   └─ 退化为普通 action 更新
      └─ 若有 action leftover:
          ├─ 当前 actions flatten 成 (F, D)
          ├─ 调 RTCProcessor.denoise_step(...)
          └─ 修正 action chunk 的 flow step
```

这里修正的是：

- **未来 action chunk**

这里使用的数据是：

- `left_over.action`

这里**不直接用**：

- `left_over.state`

---

## 4. 一次 guided infer 内部的顺序

当 `fbfm_enabled=True` 时，服务端走 `_infer_guided()`，顺序固定为：

```text
_infer_guided()
  1. 初始化 latents
  2. 初始化 actions
  3. 先跑 video loop
     └─ 每一步可能调用 _guided_video_step()
  4. 再跑 action loop
     └─ 每一步可能调用 _guided_action_step()
  5. 保存 last_generated_action_chunk
  6. 返回 postprocess 后的 action
  7. 在 infer() 末尾重新准备 left_over.action，供下一轮使用
```

所以当前实现中：

- 先生成未来一段 `video/state latent`
- 再生成未来一段 `action chunk`

不是联合同时生成。

---

## 5. 关键传参路径

### A. `feedback=True` 的传参

客户端：

```python
obs = format_obs(TASK_ENV.get_obs(), prompt)
model.infer(dict(obs=obs, feedback=True))
```

服务端：

```python
infer() -> _feedback(obs)
        -> _encode_obs(obs)
        -> left_over.append_state_latent(...)
```

结果：

- `left_over.state` 增长

---

### B. `compute_kv_cache=True` 的传参

客户端：

```python
model.infer(
    dict(
        obs=key_frame_list,
        compute_kv_cache=True,
        prev_action=action,
    )
)
```

服务端：

```python
infer() -> _compute_kv_cache(obs)
        -> obs.get('prev_action', obs.get('state'))
        -> preprocess_action(prev_action)
        -> _prepare_latent_input(...)
        -> transformer(update_cache=2)
```

结果：

- 更新 transformer cache

---

### C. 普通 infer 的传参

客户端：

```python
model.infer(dict(obs=first_obs, prompt=prompt, ...))
```

服务端：

```python
infer()
  ├─ _infer()           if fbfm_enabled=False
  └─ _infer_guided()    if fbfm_enabled=True
```

结果：

- 返回 action chunk
- 若 `fbfm_enabled=True`，还会把本轮 action 写入 `left_over.action`

---

## 6. 当前实现的真实含义

当前代码实现的不是：

- “联合 `[state; action]` 单变量 flow-matching 去噪”

当前代码实现的是：

1. **状态反馈修正 video/state latent 生成**
2. **RTC leftover 修正 action chunk 生成**

二者是两条分开的推理期修正链，共享同一个 `left_over` 缓冲，但分别消费不同字段：

- `video loop` 消费 `left_over.state`
- `action loop` 消费 `left_over.action`

