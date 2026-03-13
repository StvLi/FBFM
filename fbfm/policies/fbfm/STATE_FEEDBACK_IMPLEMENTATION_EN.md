# RTC State Feedback Extension: Implementation Notes

This document describes the **VA State Feedback** extension built on top of the LeRobot RTC baseline. In addition to the original RTC mechanism—using leftover actions from the previous chunk to constrain the next chunk's flow matching—this extension uses **the robot's truly observed states during execution of the previous chunk's remainder** (VAE-encoded image latents) to constrain the VA model's **state chunk** generation. This introduces intra-chunk state feedback, improving robustness to environmental deviations.

---

## 1. Goals and Conventions

- **State definition**: State refers to the **latent produced by a VAE encoder from images**, not proprioceptive signals such as joint angles.
- **Train-free**: No model parameters or architectures are modified; the extension lives entirely in the inference-time RTC data flow and weight computation.
- **Whole-x guidance**: State and action are treated as **one concatenated vector x**. A single backward pass computes the correction, after which separate guidance caps are applied to the state and action segments.
- **Target models**: Models that have **explicit state prediction jointly flow-matched with actions** (e.g. LingbotVA, DreamZero), where each timestep in a chunk has a fixed dimensionality of `chunk_state_dim + chunk_action_dim`.

---

## 2. Configuration (`configuration_rtc.py`)

### 2.1 New / state-feedback-related fields

| Field | Type | Description |
|-------|------|-------------|
| `state_feedback_enabled` | bool | Enable state feedback. Default: `False`. |
| `state_execution_horizon` | int | Horizon cap for the state prefix weights. |
| `state_max_guidance_weight` | float | Upper bound on the guidance scale applied to the state segment's correction. |
| `chunk_state_dim` | int \| None | **Fixed state dimensionality per timestep** inside a chunk (independent of the VAE latent dim). Required when `state_feedback_enabled`. |
| `chunk_action_dim` | int \| None | **Fixed action dimensionality per timestep** inside a chunk. Required when `state_feedback_enabled`. Must satisfy `chunk_state_dim + chunk_action_dim == x_t.shape[-1]`. |
| `state_latent_dim` | int \| None | Dimension of the observed state latent (e.g. VAE output). Used for validation; does **not** determine the chunk's D split. |

### 2.2 Validation

When `state_feedback_enabled=True`, `chunk_state_dim`, `chunk_action_dim`, and `state_max_guidance_weight` must all be positive.

---

## 3. Data Structures and Entry Points (`modeling_rtc.py`)

### 3.1 `RTCPrevChunk`

A typed container for the previous chunk's leftovers and execution-time observations, passed to `denoise_step`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `action` | Tensor \| None | Unexecuted actions from the previous chunk. Shape `(T_prev, action_dim)` or `(B, T_prev, action_dim)`. |
| `state` | Tensor \| None | State latents (VAE-encoded images) observed while executing the previous chunk's remainder. **Starts as `None`**; filled incrementally during execution. |
| `inference_delay` | int | Number of inference-delay timesteps. |
| `execution_horizon` | int \| None | Per-instance action horizon override. |
| `state_execution_horizon` | int \| None | Per-instance state horizon override. |

**Method**

- **`append_state_latent(self, new_latent: Tensor)`**  
  Called incrementally during execution: each time a new image frame is VAE-encoded, pass the resulting latent here. Accepted shapes: `(state_dim,)`, `(1, state_dim)`, or `(T_new, state_dim)`. Internally stored as 2D `(T, state_dim)` and concatenated along `dim=0`.

### 3.2 `prepare_prev_chunk_left_over(...)`

A helper to construct the `RTCPrevChunk` on the **execution side**, aligned with the ongoing action execution.

- **Arguments**: `action_left_over`, `observed_state_latents` (optional), `inference_delay`, `execution_horizon`, `state_execution_horizon`.
- **Behavior**:
  - Instantiates `RTCPrevChunk` with the given `action_left_over`; **`state` starts as `None`**.
  - If `observed_state_latents` is provided (pre-collected sequence), it is written via `append_state_latent` for consistency with the incremental path.
- **Returns**: `RTCPrevChunk`, or `None` when both arguments are `None`.

---

## 4. Chunk Layout and the D Dimension

- **Convention**: Each timestep `t`'s vector is **state dims first, action dims second**:  
  `x_t[t] = [ state(t): chunk_state_dim dims | action(t): chunk_action_dim dims ]`  
  Thus `x_t.shape = (B, T, D)` with `D = chunk_state_dim + chunk_action_dim`.
- **D is fixed by configuration**, not derived from `state_latent_dim`. `denoise_step` validates `chunk_state_dim + chunk_action_dim == x_t.shape[-1]` at runtime.

---

## 5. `denoise_step` Extension Logic

### 5.1 Input types

`prev_chunk_left_over` accepts:
- **`Tensor`** – action-only (backward-compatible with original RTC);
- **`RTCPrevChunk`** – joint state+action when state feedback is active;
- **`None`** – no guidance; returns raw `v_t`.

### 5.2 Building the combined prefix (state feedback path)

A `combined_prefix` of shape `(B, T, D)` is built with state and action segments laid out per timestep:
- **State segment `[:chunk_state_dim]`**: filled from `RTCPrevChunk.state` for the first `t_state` steps; zeros elsewhere.
- **Action segment `[chunk_state_dim:]`**: filled from `RTCPrevChunk.action` for the first `t_act` steps (if available); zeros elsewhere.

`prev_chunk_tensor` is then set to `combined_prefix` and fed into the same error and gradient computation as in the original RTC.

### 5.3 Weights `diag(W)` (hard-edge mask)

**Structure** (matching the design diagram):  
**[1s for available state steps; 0s for remaining state steps; 1s for leftover action steps; 0s for remaining action steps]**

Per `(t, dim)`:
- State dims: `1` if `t < t_state_cap`, else `0`.  
  `t_state_cap = min(t_state, state_execution_horizon, T)`; grows as `append_state_latent` is called.
- Action dims: `1` if `t < t_act`, else `0`.

**Implementation** — `weights` shape `(1, T, D)`:
```python
weights[:, :t_state_cap, :chunk_state_dim] = 1.0
weights[:, :t_act,       chunk_state_dim:] = 1.0
```
All other entries remain `0`. Because `weights` is recomputed each `denoise_step` call from the current `RTCPrevChunk.state` length, it **automatically tracks incremental `append_state_latent` calls**.

### 5.4 Single backward pass over the whole x

```python
err = (prev_chunk_tensor - x1_t) * weights   # x1_t = x_t - time * v_t
correction = torch.autograd.grad(x1_t, x_t, err.detach())[0]
```

State and action errors are merged in one `err` tensor; a **single** `autograd.grad` call produces the correction for the entire x.

### 5.5 Guidance weights (symmetric capping)

- **`base_guidance_weight`**: computed from τ; shared by both segments.
- **Symmetric caps**:
  - `action_max_guidance_weight` ← `rtc_config.max_guidance_weight`
  - `state_max_guidance_weight` ← `rtc_config.state_max_guidance_weight`
  - `action_guidance_weight = min(base_guidance_weight, action_max_guidance_weight)`
  - `state_guidance_weight  = min(base_guidance_weight, state_max_guidance_weight)`
- **Result**:
  - *State feedback active*:  
    `result = v_t - cat(state_guidance_weight * correction_state, action_guidance_weight * correction_action, dim=-1)`
  - *Otherwise*:  
    `result = v_t - action_guidance_weight * correction`

---

## 6. Recommended Execution-Side Workflow

1. Before requesting the next chunk, call  
   `rtc_prev_chunk = prepare_prev_chunk_left_over(action_left_over=..., observed_state_latents=None, ...)`.  
   `state` is `None` at this point.
2. While executing the remaining actions from the previous chunk, call  
   `rtc_prev_chunk.append_state_latent(vae_latent)` each time a new image frame is encoded.
3. Pass `rtc_prev_chunk` to the policy:  
   `policy.predict_action_chunk(..., prev_chunk_left_over=rtc_prev_chunk, ...)`.  
   The policy forwards it to `RTCProcessor.denoise_step` internally.
4. Set `state_feedback_enabled=True` in `RTCConfig` and configure `chunk_state_dim`, `chunk_action_dim` (plus optional `state_execution_horizon`, `state_max_guidance_weight`).

---

## 7. Files and Dependencies

| File | Role |
|------|------|
| `configuration_rtc.py` | RTC and state feedback configuration fields and validation. |
| `modeling_rtc.py` | `RTCPrevChunk`, `prepare_prev_chunk_left_over`, `RTCProcessor.denoise_step` (combined prefix, `diag(W)`, single backward, symmetric guidance capping). |
| `action_queue.py` | Action queue (`get_left_over()` provides the action leftover); no direct dependency on state feedback. |
| `debug_tracker.py` / `debug_visualizer.py` | Optional debug recording and visualization. |

On the policy side (e.g. LingbotVA / DreamZero), the denoise loop must forward `prev_chunk_left_over` and `inference_delay` to `denoise_step`, and the model must output a joint `[state; action]` chunk whose last dimension equals `chunk_state_dim + chunk_action_dim`.
