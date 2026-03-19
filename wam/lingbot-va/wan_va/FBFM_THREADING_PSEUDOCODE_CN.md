# FBFM-Threading 伪代码

目标：在不改训练的前提下，将 `Lingbot-VA` 的两段式推理改成“执行与推理并发”：

- `video loop`：用执行中返回的 `obs` 形成 `state feedback`
- `action loop`：用上一 chunk 的 `action leftover` 做 RTC 修正

## 1. 共享状态

```python
class ActionItem:
    action
    chunk_id

class SharedState:
    action_queue              # deque[ActionItem]，唯一的 action 真值来源
    state_feedback_buffer     # 最近 K 步 feedback latent
    feedback_cursor           # 推理线程上次消费到的位置

    latest_obs                # 最近一次真实 observation
    obs_ready                 # 首帧 observation 是否已就绪

    last_generated_action     # 最近一次生成出的完整 action chunk，仅用于 debug

    cur_chunk_id              # 最近一次生成的 chunk id
    exec_chunk_id             # 当前执行中的 chunk id
    steps_in_cur_chunk        # 当前 chunk 已执行步数

    frame_st_id               # cache / grid id 对应的 frame 偏移

    need_new_chunk            # 是否需要后台推理新 chunk
    inference_in_flight       # 当前是否已有后台推理在运行
    stop_flag                 # 退出标志

    low_watermark             # 队列低水位
    high_watermark            # 队列高水位

    mutex
    cond
```

约定：

- `action_leftover` 不单独保存，始终由 `action_queue` 推导
- `state_feedback_buffer` 只保留最近 `K` 步

## 2. 启动

```python
procedure INITIALIZE(prompt, init_obs):
    with mutex:
        current_prompt = prompt
        latest_obs = init_obs
        obs_ready = (init_obs is not None)

        action_queue = empty deque
        state_feedback_buffer = empty deque
        feedback_cursor = 0

        last_generated_action = None
        cur_chunk_id = 0
        exec_chunk_id = None
        steps_in_cur_chunk = 0
        frame_st_id = 0

        need_new_chunk = True      # 避免初始死锁
        inference_in_flight = False
        stop_flag = False

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
            frame_st_id_snapshot = frame_st_id
            feedback_states = state_feedback_buffer[feedback_cursor:]
            leftover_action = build_leftover_from_queue(action_queue)

            next_chunk_id = cur_chunk_id + 1
            need_new_chunk = False
            inference_in_flight = True

        action_chunk = GUIDED_INFERENCE(
            obs=obs,
            feedback_states=feedback_states,
            leftover_action=leftover_action,
            frame_st_id=frame_st_id_snapshot,
        )

        actions = split_action_chunk(action_chunk)

        with mutex:
            accepted = len(action_queue) < high_watermark

            if accepted:
                action_queue.extend(
                    [ActionItem(action=a, chunk_id=next_chunk_id) for a in actions]
                )
                last_generated_action = action_chunk
                cur_chunk_id = next_chunk_id

                # 只有 chunk 被接纳，才推进 feedback_cursor
                feedback_cursor = len(state_feedback_buffer)

            inference_in_flight = False
            cond.notify_all()
```

## 4. 执行线程

```python
procedure EXECUTION_LOOP():
    while True:
        with mutex:
            while len(action_queue) == 0 and not stop_flag:
                cond.wait()
            if stop_flag:
                break

            item = action_queue.popleft()
            action = item.action

            if exec_chunk_id != item.chunk_id:
                exec_chunk_id = item.chunk_id
                steps_in_cur_chunk = 0

            steps_in_cur_chunk += 1

        execute_action(action)
        obs = get_observation()
        state_latent = encode_obs(obs)

        with mutex:
            latest_obs = obs
            obs_ready = True

            state_feedback_buffer.append(state_latent)
            state_feedback_buffer = trim_to_last_K(state_feedback_buffer)
            feedback_cursor = min(feedback_cursor, len(state_feedback_buffer))

            frame_st_id += 1

            if should_trigger_new_chunk(
                queue_len=len(action_queue),
                steps_in_cur_chunk=steps_in_cur_chunk,
                refill_threshold=s_chunk,
                low_watermark=low_watermark,
            ):
                need_new_chunk = True
                cond.notify_all()
```

## 5. 推理主过程

```python
function GUIDED_INFERENCE(obs, feedback_states, leftover_action, frame_st_id):
    init_latent = encode_obs(obs)

    latents = randn_video_latents()
    for t in video_timesteps:
        latents = VIDEO_STEP(
            latents=latents,
            feedback_states=feedback_states,
            timestep=t,
            frame_st_id=frame_st_id,
            latent_cond=init_latent,
        )

    actions = randn_action_chunk()
    for t in action_timesteps:
        actions = ACTION_STEP(
            actions=actions,
            leftover_action=leftover_action,
            timestep=t,
            frame_st_id=frame_st_id,
        )

    return actions
```

## 6. `video loop`

```python
function VIDEO_STEP(latents, feedback_states, timestep, frame_st_id, latent_cond):
    if feedback_states is empty:
        return VIDEO_BASE_STEP(latents, timestep, frame_st_id, latent_cond)

    x_t = flatten_video_latents(latents)
    y = format_feedback_states(feedback_states)
    W = build_state_weights(feedback_states)

    def denoise_fn(x):
        latent_input = unflatten_video_latents(x)
        return video_denoise(latent_input, timestep, frame_st_id, latent_cond)

    return scheduler.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=latents,
        constrained_y=y,
        weights=W,
    )
```

## 7. `action loop`

```python
function ACTION_STEP(actions, leftover_action, timestep, frame_st_id):
    if leftover_action is empty:
        return ACTION_BASE_STEP(actions, timestep, frame_st_id)

    x_t = flatten_action_chunk(actions)
    y = format_leftover_action(leftover_action)
    W = build_action_weights(leftover_action)

    def denoise_fn(x):
        action_input = unflatten_action_chunk(x)
        return action_denoise(action_input, timestep, frame_st_id)

    return action_scheduler.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=actions,
        constrained_y=y,
        weights=W,
    )
```

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

- `action_queue`
- `state_feedback_buffer`
- `feedback_cursor`
- `latest_obs`
- `obs_ready`
- `cur_chunk_id / exec_chunk_id`
- `steps_in_cur_chunk`
- `frame_st_id`
- `need_new_chunk`
- `inference_in_flight`
- `stop_flag`

锁外：

- `execute_action`
- `get_observation`
- `encode_obs`
- `video loop / action loop`
- `scheduler.step`

## 10. 一句话总结

执行线程持续“执行动作并回传 `obs`”，推理线程持续“读取未消费的 `state feedback` 和当前 `action leftover`，先修正 `video loop`，再修正 `action loop`，并按水位补充 `action_queue`”。

