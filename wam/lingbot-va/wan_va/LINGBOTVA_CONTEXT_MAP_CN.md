# Lingbot-VA 中“上下文”对应的变量

这里的“上下文”指：为下一次推理准备的、描述“刚刚发生了什么”的信息。

## 1. 观测上下文

- `obs`
  当前真实 observation。

- `latent_model_input`
  `obs` 经 `_encode_obs(obs)` 后得到的 latent 表示。

- `init_latent`
  `_infer()` 开始时由当前 observation 编码得到，作为本轮推理的起始条件。

- `prev_chunk_left_over.states`
  当前轮执行过程中累计的 observation latent，作为 `video loop` 的 state feedback 前缀。

## 2. 动作上下文

- `action_model_input`
  由 `preprocess_action(...)` 得到的动作表示。

- `prev_chunk_left_over.actions`
  上一 chunk 未执行完的 action 前缀，作为 `action loop` 的 RTC 约束。

## 3. 时间上下文

- `frame_st_id`
  当前 chunk 在 frame 时间轴上的起点偏移。

- `grid_id`
  由 `get_mesh_id(...)` 生成，带着 `frame_st_id` 进入模型。

- `timesteps`
  当前 flow/diffusion 步的时间参数。

## 4. 模型内部上下文

- transformer KV cache
  由：
  - `create_empty_cache(...)`
  - `update_cache=1`
  - `update_cache=2`
  - `_compute_kv_cache()`
  维护。

作用：

- 让下一次推理能够读取已经发生过的 observation / action 历史，而不是从空白状态开始。

## 5. 推理期修正上下文

- `prev_chunk_left_over.get_constrained_states()`
  当前轮 `video loop` 要参考的 feedback state。

- `prev_chunk_left_over.get_state_prefix_weights()`
  上述 state 中哪些时间步有效。

- `prev_chunk_left_over.get_constrained_actions()`
  当前轮 `action loop` 要参考的 leftover action。

- `prev_chunk_left_over.get_action_prefix_weights()`
  上述 action 中哪些时间步有效。

## 6. 一句话总结

在当前代码里，“为下一次推理准备上下文”就是把：

- `obs -> latent_model_input`
- `action -> action_model_input`
- `frame_st_id -> grid_id`
- `feedback / leftover -> prev_chunk_left_over`
- 历史 token -> KV cache`

这些信息准备好，让下一次 `_infer()` 能从正确的历史继续生成。

