# FBFM.PrevChunk Class Detailed Guide

## 1. Overview

The `PrevChunk` class is the core state manager within the **FBFM (Flow-Based Feedback Model)** strategy. It is primarily used to maintain the action sequences (Actions) and observed state latents from the previous round during multi-chunk generation processes. This provides continuity constraints and feedback correction for the current chunk's generation.

This class implements the design philosophy of "Feedback as Constraint." By encapsulating historical data into weighted tensors, it guides the Diffusion Model or Flow Matching Model to converge towards the desired state trajectory during the denoising process.

## 2. Core Design Intent

In Real-Time Control scenarios, models typically perform rolling predictions in fixed-length chunks. To ensure action smoothness and state consistency, `PrevChunk` assumes the following responsibilities:

1.  **State Caching**: Stores action leftovers from the previous round that were not executed and the real states (Observed States) observed during execution.
2.  **Constraint Injection**: Dynamically generates weight masks based on the configured `constrain_mode`, instructing the Scheduler which timesteps require strong constraints and which can be freely generated.
3.  **Dimension Alignment**: Ensures input state and action tensors match the model's expected fixed dimensions `(T, D)`, handling shape matching during dynamic appending.

## 3. Class Structure and Attributes

### 3.1 Initialization Parameters (`__init__`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `constrain_mode` | str | `"Feedback"` | Constraint mode. Options:<br>- `"Feedback"`: Constrains both action prefixes and state prefixes.<br>- `"RTC"`: Constrains only action prefixes (Standard RTC mode).<br>- `"None"`: No constraints, fully free generation. |
| `actions` | Tensor | `None` | Initial action tensor, shape should be `(T_a, D_a)`. If provided, fills the internal buffer. |
| `action_constrained_num` | int | `0` | Initial number of action steps to constrain. |
| `action_num` | int | `16` | Maximum time steps for the action buffer ($T_a$). |
| `action_dim` | int | `16` | Dimension of the action space ($D_a$). |
| `states` | Tensor | `None` | Initial state tensor, shape should be `(T_s, D_s)`. |
| `state_constrained_num` | int | `0` | Initial number of state steps to constrain. |
| `state_num` | int | `4` | Maximum time steps for the state buffer ($T_s$). |
| `state_dim` | int | `128` | Dimension of the state latent variable ($D_s$), usually determined by the VAE encoder output. |
| `inference_delay` | int | `0` | Inference delay steps, used for time axis alignment compensation. |

### 3.2 Core Attributes

-   `self.actions`: A fixed-size tensor of shape `(action_num, action_dim)` storing action history. Unfilled parts are zero.
-   `self.action_constrained_num`: An integer indicating how many leading steps in `self.actions` are valid constraint data.
-   `self.states`: A fixed-size tensor of shape `(state_num, state_dim)` storing state history.
-   `self.state_constrained_num`: An integer indicating how many leading steps in `self.states` are valid observed data.
-   `self.constrain_mode`: The current constraint strategy identifier.

## 4. Key Method Details

### 4.1 `append_new_state(new_state: Tensor)`
**Function**: Appends a newly observed state to the state buffer.
**Logic**:
1.  **Dimension Normalization**: Automatically converts 1D input `(D,)` to 2D `(1, D)`.
2.  **Dimension Check**: Verifies if the input dimension matches the `state_dim` from initialization.
3.  **Sequential Filling**:
    -   If the buffer is not full (`state_constrained_num < state_num`), writes the new state to index `state_constrained_num` and increments the counter.
    -   If the buffer is full, the current implementation chooses to **discard** the new state to maintain historical order (can be modified to ring-buffer overwrite based on requirements).

### 4.2 `get_action_prefix_weights() -> Tensor`
**Function**: Generates the weight mask for the action prefix.
**Returns**: A tensor of shape `(action_num,)`.
**Rules**:
-   If `constrain_mode` is `"RTC"` or `"Feedback"`: The first `action_constrained_num` elements are `1.0`, others are `0.0`.
-   If `constrain_mode` is `"None"`: All `0.0`.

### 4.3 `get_state_prefix_weights() -> Tensor`
**Function**: Generates the weight mask for the state prefix.
**Returns**: A tensor of shape `(state_num,)`.
**Rules**:
-   Only when `constrain_mode` is `"Feedback"`: The first `state_constrained_num` elements are `1.0`, others are `0.0`.
-   Other modes (`"RTC"`, `"None"`): All `0.0` (since standard RTC does not directly constrain state latents, only actions).

### 4.4 `get_prefix_weights() -> Tensor`
**Function**: Retrieves the complete combined weight vector for direct use by the Scheduler.
**Returns**: A tensor of shape `(action_num + state_num,)`.
**Structure**: `[Action Weight Part, State Weight Part]`.
**Usage**: This tensor is passed directly to `WrapperedFlowMatchScheduler` to calculate the $weights$ in the gradient correction term $(y_{target} - x_1) \cdot weights$.

### 4.5 Data Access Methods
-   `get_constrained_actions()`: Returns the current `self.actions` tensor (including zero-padded parts).
-   `get_constrained_states()`: Returns the current `self.states` tensor.
-   `get_constrain_mode()`: Returns the current mode string.

## 5. Workflow Diagram

The following diagram illustrates the data flow and control flow of the `PrevChunk` class during the multi-chunk rolling generation process. The core logic lies in updating the internal buffer with real observed states from the previous round and generating weight masks to guide the diffusion denoising process of the current round.