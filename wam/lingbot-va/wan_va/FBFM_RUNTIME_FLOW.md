# FBFM Runtime Flow In Current Lingbot-VA

This document describes the **actual runtime flow in the current codebase**.
It does not describe the idealized joint `[state; action]` formulation. It
only explains what the code currently does.

Relevant files:

- [wan_va_server.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/wan_va/wan_va_server.py)
- [eval_polict_client_openpi.py](/c:/Users/zhang/Documents/GitHub/FBFM/wam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py)
- [modeling_rtc_fbfm.py](/c:/Users/zhang/Documents/GitHub/FBFM/fbfm/policies/fbfm/modeling_rtc_fbfm.py)

## Summary

There are two separate inference lines in the current implementation:

1. `video loop`
   Uses **state feedback** from real observations to guide future video latent generation.
2. `action loop`
   Uses **RTC-style previous action leftover** to guide future action chunk generation.

The current code does **not** jointly denoise `[state; action]` in one flow variable.

## Main Objects

### Client-side requests

The evaluation client sends three kinds of requests to the server:

1. `reset`
2. `feedback`
3. `compute_kv_cache`
4. normal `infer`

### Server-side persistent state

The server keeps these fields across requests:

- `self.left_over`
  Type: `RTCPrevChunk`
  Meaning:
  - `left_over.state`: collected state feedback latents from real observations
  - `left_over.action`: flattened action chunk from the last generated chunk
- `self.last_generated_action_chunk`
  The most recent generated normalized action tensor before postprocess
- `self.frame_st_id`
  Running frame offset used by cache/grid ids
- `self.init_latent`
  Initial encoded observation latent

## Request Flow

### 1. Reset Request

Client sends:

```python
model.infer(dict(reset=True, prompt=prompt, ...))
```

Server entry:

- `VA_Server.infer()`

Server action:

- calls `_reset(prompt=prompt)`

Effects:

- clears `frame_st_id`
- clears `init_latent`
- clears `left_over`
- clears `last_generated_action_chunk`
- clears transformer / VAE caches

## 2. Normal Infer Request

Client sends:

```python
model.infer(dict(obs=first_obs, prompt=prompt, ...))
```

Server entry:

- `VA_Server.infer()`

Branching:

- if `fbfm_enabled=True`, calls `_infer_guided(obs, frame_st_id=self.frame_st_id)`
- else, calls `_infer(obs, frame_st_id=self.frame_st_id)`

### Parameter path

Client payload:

- `obs`
- `prompt`
- optional other visualization flags

Server uses:

- `obs['obs']` inside `_encode_obs()`
- `prompt` only during reset for prompt embedding setup

## 3. Feedback Request

Client sends feedback while executing the current action chunk:

```python
obs = format_obs(TASK_ENV.get_obs(), prompt)
model.infer(dict(obs=obs, feedback=True))
```

Source of feedback observation:

- `TASK_ENV.get_obs()`

Client formatting function:

- `format_obs(...)` in `eval_polict_client_openpi.py`

Produced fields:

- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`
- `observation.state`
- `task`

Server entry:

- `VA_Server.infer()`
- branch `feedback=True`
- calls `_feedback(obs=obs)`

Server action inside `_feedback()`:

1. calls `_encode_obs(obs)`
2. converts observation images to video latent
3. if `self.left_over` is `None`, creates it via:

```python
FBFM.prepare_prev_chunk_left_over(
    inference_delay=self.fbfm_inference_delay,
    execution_horizon=self.rtc_processor.rtc_config.execution_horizon,
)
```

4. appends encoded latent into:

```python
self.left_over.append_state_latent(...)
```

### Meaning

This request only updates:

- `left_over.state`

It does not generate actions.

## 4. Compute KV Cache Request

Client sends:

```python
model.infer(
    dict(
        obs=key_frame_list,
        compute_kv_cache=True,
        prev_action=action,
        ...
    )
)
```

Server entry:

- `VA_Server.infer()`
- branch `compute_kv_cache=True`
- calls `_compute_kv_cache(obs)`

### Parameter path

Client payload:

- `obs`
- `prev_action`

Server extraction:

```python
prev_action = obs.get('prev_action', obs.get('state'))
```

Then:

1. `latent_model_input = self._encode_obs(obs)`
2. `action_model_input = self.preprocess_action(prev_action)`
3. both are packed by `_prepare_latent_input(...)`
4. server calls transformer twice with `update_cache=2`
   - once with `action_mode=False`
   - once with `action_mode=True`

### Meaning

This request updates the transformer's cache using:

- real observation latent
- previous action chunk

It does not directly append to `left_over`.

## Two Inference Lines

## A. Video Line: State Feedback Guidance

Code path:

- `_infer_guided()`
- `_guided_video_step()`
- `_predict_video_noise()`

### Data source

The guidance source is:

```python
self.left_over.state
```

This state data comes from prior `feedback=True` requests.

### What is guided

Current guided variable:

- `latents`

Shape before flatten:

- `(B, C, F, H, W)`

Flattened for RTC guidance:

```python
rearrange(latents, 'b c f h w -> f (c h w)')
```

### Parameter passing chain

1. `_infer_guided()` calls:

```python
latents = self._guided_video_step(latents, t, sigma_time, frame_st_id, latent_cond)
```

2. `_guided_video_step()` sets:

```python
prefix = self.left_over.state if self.left_over is not None else None
```

3. If no prefix:

- falls back to `_predict_video_noise(...)`
- then `self.scheduler.step(...)`

4. If prefix exists:

- flattens current `latents`
- defines `original_denoise(flat_latents)`
- passes both into:

```python
self.rtc_processor.denoise_step(
    x_t=flat_latents,
    prev_chunk_left_over=prefix,
    inference_delay=self.fbfm_inference_delay,
    time=sigma_time,
    original_denoise_step_partial=original_denoise,
    execution_horizon=self.rtc_processor.rtc_config.execution_horizon,
)
```

5. result is unflattened back into latent tensor
6. updated by `self.scheduler.step(...)`

### Important point

In the current code, the video line only uses:

- `left_over.state`

It does not use `left_over.action` for video guidance.

## B. Action Line: RTC Previous Action Leftover Guidance

Code path:

- `_infer_guided()`
- `_guided_action_step()`
- `_predict_action_noise()`

### Data source

The guidance source is:

```python
self.left_over.action
```

This action data is prepared after each normal infer returns.

### How `left_over.action` is created

After guided infer finishes in `VA_Server.infer()`:

```python
self.left_over = FBFM.prepare_prev_chunk_left_over(
    action_left_over=self._flatten_action_chunk_for_guidance(self.last_generated_action_chunk),
    inference_delay=self.fbfm_inference_delay,
    execution_horizon=self.rtc_processor.rtc_config.execution_horizon,
)
```

So:

- current generated action chunk
- is flattened
- then becomes next round's `left_over.action`

### What is guided

Current guided variable:

- `actions`

Shape before flatten:

- `(B, C, F, action_per_frame, 1)`

Flattened for RTC guidance:

```python
rearrange(actions, 'b c f h w -> f (c h w)')
```

### Parameter passing chain

1. `_infer_guided()` action loop calls:

```python
actions = self._guided_action_step(actions, t, sigma_time, frame_st_id, action_cond)
```

2. `_guided_action_step()` sets:

```python
prefix = self.left_over.action if self.left_over is not None else None
```

3. If no prefix:

- falls back to `_predict_action_noise(...)`
- then `self.action_scheduler.step(...)`

4. If prefix exists:

- flattens current actions
- defines `original_denoise(flat_actions)`
- passes both into:

```python
self.rtc_processor.denoise_step(
    x_t=flat_actions,
    prev_chunk_left_over=prefix,
    inference_delay=self.fbfm_inference_delay,
    time=sigma_time,
    original_denoise_step_partial=original_denoise,
    execution_horizon=self.rtc_processor.rtc_config.execution_horizon,
)
```

5. result is unflattened back to action tensor
6. updated by `self.action_scheduler.step(...)`

### Important point

In the current code, the action line only uses:

- `left_over.action`

It does not directly use `left_over.state` to guide action flow.

## `_prepare_latent_input(...)` Call Sites

Current meaningful call patterns:

### Video branch

```python
self._prepare_latent_input(
    latent_model_input=latents,
    action_model_input=None,
    latent_t=timestep,
    action_t=timestep,
    latent_cond=latent_cond,
    action_cond=None,
    frame_st_id=frame_st_id,
)
```

Used in:

- `_predict_video_noise(...)`

### Action branch

```python
self._prepare_latent_input(
    latent_model_input=None,
    action_model_input=actions,
    latent_t=t,
    action_t=t,
    latent_cond=None,
    action_cond=action_cond,
    frame_st_id=frame_st_id,
)
```

Used in:

- `_predict_action_noise(...)`

### Cache update branch

```python
self._prepare_latent_input(
    latent_model_input=latent_model_input,
    action_model_input=action_model_input,
    frame_st_id=self.frame_st_id,
)
```

Used in:

- `_compute_kv_cache(...)`

## What The Current Code Does Not Do

The current code does not do the following:

- it does not jointly denoise `[state; action]` in one variable
- it does not pass `left_over.state` directly into the action guidance equation
- it does not pass `left_over.action` directly into the video guidance equation

Instead, it implements:

1. state-feedback-guided video latent generation
2. RTC-guided action chunk generation

as two separate inference-time correction lines.
