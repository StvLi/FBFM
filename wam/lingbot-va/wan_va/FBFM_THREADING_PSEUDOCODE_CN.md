# FBFM-Threading 伪代码

目标：在不改训练的前提下，将 `Lingbot-VA` 的两段式推理改成“执行与推理并发”：

- `video loop`：用执行中回传的 `obs` 形成 `state feedback`
- `action loop`：用上一 chunk 未执行的 `action` 做 RTC 修正

当前版本以 `FBFM.PrevChunk` 作为线程间共享的约束容器。

## 1. 共享状态

```python
class SharedState:
    A_queue                  # 待执行动作队列
    prev_chunk_left_over     # FBFM.PrevChunk

    latest_obs               # 最近一次 observation
    obs_ready                # 首帧 observation 是否已就绪

    need_new_chunk           # 是否需要生成新 chunk
    inference_in_flight      # 当前是否已有后台推理
    stop_flag                # 退出标志

    frame_st_id              # frame 级时间偏移
    steps_in_cur_chunk       # 当前 chunk 已执行步数

    mutex
    cond
```

约定：

- `prev_chunk_left_over.actions` 表示上一 chunk 未执行的 action 前缀
- `prev_chunk_left_over.states` 表示当前 chunk 执行过程中累计的 feedback latent
- 开始新一轮推理时，`states` 清空重新累计

## 2. 初始化

```python
procedure INITIALIZE(prompt, init_obs):
    with mutex:
        latest_obs = init_obs
        obs_ready = (init_obs is not None)

        A_queue = empty deque
        prev_chunk_left_over = None

        need_new_chunk = True
        inference_in_flight = False
        stop_flag = False

        frame_st_id = 0
        steps_in_cur_chunk = 0

        cond.notify_all()
```

## 3. 推理线程

```python
procedure INFERENCE_LOOP():
    while True:
        with mutex:
            while ((not need_new_chunk) or inference_in_flight or not obs_ready) and not stop_flag:
                cond.wait()
            if stop_flag:
                break

            obs = latest_obs
            cur_prev_chunk = prev_chunk_left_over
            cur_frame_st_id = frame_st_id

            need_new_chunk = False
            inference_in_flight = True

        action_chunk = GUIDED_INFERENCE(
            obs=obs,
            prev_chunk=cur_prev_chunk,
            frame_st_id=cur_frame_st_id,
        )

        with mutex:
            A_queue.extend(split_action_chunk(action_chunk))
            inference_in_flight = False
            cond.notify_all()
```

## 4. 执行线程

```python
procedure EXECUTION_LOOP():
    while True:
        with mutex:
            while len(A_queue) == 0 and not stop_flag:
                cond.wait()
            if stop_flag:
                break

            action = A_queue.popleft()
            steps_in_cur_chunk += 1

        execute_action(action)

        if steps_in_cur_chunk % action_per_frame == 0:
            obs = get_observation()
            state_latent = encode_obs(obs)

            with mutex:
                latest_obs = obs
                obs_ready = True
                frame_st_id += 1

                if prev_chunk_left_over is not None:
                    prev_chunk_left_over.append_new_state(state_latent)

                if should_trigger_new_chunk(
                    queue_len=len(A_queue),
                    steps_in_cur_chunk=steps_in_cur_chunk,
                    refill_threshold=s_chunk,
                    low_watermark=low_watermark,
                ):
                    need_new_chunk = True
                    cond.notify_all()
```

## 5. 推理主过程

```python
function GUIDED_INFERENCE(obs, prev_chunk, frame_st_id):
    init_latent = encode_obs(obs)

    latents = randn_video_latents()
    for t in video_timesteps:
        latents = VIDEO_STEP(
            latents=latents,
            prev_chunk=prev_chunk,
            timestep=t,
            frame_st_id=frame_st_id,
            latent_cond=init_latent,
        )

    actions = randn_action_chunk()
    for t in action_timesteps:
        actions = ACTION_STEP(
            actions=actions,
            prev_chunk=prev_chunk,
            timestep=t,
            frame_st_id=frame_st_id,
        )

    # 下一轮推理前更新 PrevChunk:
    # 1. actions = 本轮生成后尚未执行的 action
    # 2. states 清空
    next_prev_chunk = FBFM.PrevChunk(
        constrain_mode="Feedback",
        actions=format_leftover_action(actions),
        action_constrained_num=get_leftover_len(actions),
        action_num=...,
        action_dim=...,
        states=None,
        state_constrained_num=0,
        state_num=...,
        state_dim=...,
        inference_delay=d,
    )

    with mutex:
        prev_chunk_left_over = next_prev_chunk
        steps_in_cur_chunk = 0

    return actions
```

## 6. `video loop`

```python
function VIDEO_STEP(latents, prev_chunk, timestep, frame_st_id, latent_cond):
    if prev_chunk is None:
        return VIDEO_BASE_STEP(latents, timestep, frame_st_id, latent_cond)

    x_t = latents
    y = prev_chunk.get_constrained_states()
    W = prev_chunk.get_state_prefix_weights()

    def denoise_fn(x):
        return video_denoise(x, timestep, frame_st_id, latent_cond)

    return scheduler.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=latents,
        constrained_y=y,
        weights=W,
    )
```

含义：

- `prev_chunk.states -> y`
- `prev_chunk.get_state_prefix_weights() -> W`
- 修正对象是未来 `video/state latent`

## 7. `action loop`

```python
function ACTION_STEP(actions, prev_chunk, timestep, frame_st_id):
    if prev_chunk is None:
        return ACTION_BASE_STEP(actions, timestep, frame_st_id)

    x_t = actions
    y = prev_chunk.get_constrained_actions()
    W = prev_chunk.get_action_prefix_weights()

    def denoise_fn(x):
        return action_denoise(x, timestep, frame_st_id)

    return action_scheduler.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=actions,
        constrained_y=y,
        weights=W,
    )
```

含义：

- `prev_chunk.actions -> y`
- `prev_chunk.get_action_prefix_weights() -> W`
- 修正对象是未来 `action chunk`

## 8. 触发条件

```python
function should_trigger_new_chunk(queue_len, steps_in_cur_chunk, refill_threshold, low_watermark):
    if queue_len < low_watermark:
        return True
    if steps_in_cur_chunk >= refill_threshold:
        return True
    return False
```

## 9. 锁保护范围

锁内：

- `A_queue`
- `prev_chunk_left_over`
- `latest_obs`
- `obs_ready`
- `need_new_chunk`
- `inference_in_flight`
- `stop_flag`
- `frame_st_id`
- `steps_in_cur_chunk`

锁外：

- `execute_action`
- `get_observation`
- `encode_obs`
- `video loop / action loop`
- `scheduler.step`

## 10. 一句话总结

执行线程持续“执行动作并将 observation 追加到 `PrevChunk.states`”，推理线程持续“读取 `PrevChunk`，在 `video loop` 中用 `states` 做 feedback，在 `action loop` 中用 `actions` 做 RTC，并在新一轮推理开始前整体更新 `PrevChunk`”。

