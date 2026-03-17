"""
algo/fbfm.py — Feed-Back Flow Matching (FBFM) rollout

FBFM rollout rhythm (single-thread simulation of async dual-thread):
    Large cycle (fine-grained interleaving):
        for each denoising "inner block" of n_inner steps:
            [Inference chain]  Run n_inner denoising steps with current feedback
            [Execution chain]  Execute 1 action, read back (state, action)
                               → feed (state, action) back to inference chain
        → next large cycle

Key innovation over RTC: the execution chain and inference chain interact
at every single action step. Each executed action immediately updates the
flow field, enabling within-chunk correction when the environment changes.

Reference: Thoughts.md §实验组, Algorithm: FBFM pseudo-code

Shapes throughout:
    obs:     (obs_dim,)          = (2,)     — [x, x_dot]
    X^τ:     (H, state_dim + action_dim) or (H, action_dim)
             — see note below on X definition
    W:       (H,)                           — mask weights
    chunk:   (H, action_dim)     = (16, 1)  — output action sequence

NOTE on X definition in FBFM:
    Thoughts.md §FBFM says X^τ_t = [Z^τ_t, A^τ_t], i.e. the state latent
    and action are concatenated. For this MSD toy model we simplify:
        X^τ = A^τ  (action chunk only, shape (H, action_dim))
    because the state is directly observable (no VAE encoder needed).
    The obs conditioning already carries the state information.
    If you want to extend to the full formulation, change action_dim to
    state_dim + action_dim and adjust the output head in dit.py accordingly.
"""

import numpy as np
import torch
from model.dit import FlowMatchingDiT


# -----------------------------------------------------------------------
# [USER IMPLEMENTS] Core FBFM guided inference
# -----------------------------------------------------------------------

def guided_inference_fbfm(
    policy: FlowMatchingDiT,
    obs: np.ndarray,            # (obs_dim,) = (2,)  current observation
    X_prev: np.ndarray,         # (H, action_dim) = (16, 1)  previous chunk,
                                #   right-padded with zeros to length H
    W: np.ndarray,              # (H,)  mask weights: 1.0 for real, 0.0 for padding
    n_steps: int = 16,          # total denoising steps
    n_inner: int = 4,           # denoising steps per interleave block
    beta: float = 10.0,         # max guidance weight β
    env_feedback_fn=None,       # Callable: () -> (obs_new, action_executed)
                                #   called once per interleave block (after n_inner steps)
                                #   returns the latest env observation and the action just executed
    device: str = "cpu",
) -> np.ndarray:                # (H, action_dim) = (16, 1)  new action chunk
    """
    FBFM guided Flow Matching inference with real-time execution feedback.

    The denoising loop is interleaved with execution:
        for block in range(n_steps // n_inner):
            run n_inner denoising steps  (inference chain)
            call env_feedback_fn()       (execution chain: execute 1 action)
            update Y and obs from feedback

    ── Algorithm (one denoising step at noise level τ) ──────────────────
    Given:
        X^τ  : (H, action_dim)  current noisy sample
        obs  : (obs_dim,)       conditioning observation (updated each block)
        τ    : scalar           current noise level ∈ [0, 1]
        Y    : (H, action_dim)  target = X_prev (right-padded, masked by W)
                                updated after each execution step

    1. Predict velocity:
           v = policy(X^τ, obs, τ)          # (H, action_dim)

    2. Predict clean sample:
           X̂¹_t = X^τ + (1 - τ) * v         # (H, action_dim)

    3. Compute guidance gain:
           r²_τ = (1-τ)² / (τ² + (1-τ)²)
           k_p  = min(β, (1-τ) / (τ * r²_τ))   # clamp τ away from 0

    4. Compute guidance gradient (Jacobian of X̂¹ w.r.t. X^τ):
           ∂X̂¹/∂X^τ ≈ I + (1-τ) * ∂v/∂X^τ    # requires torch.autograd

    5. Compute error weighted by mask:
           e = (Y - X̂¹)^T diag(W)             # (H, action_dim), masked

    6. Guided velocity:
           v_FBFM = v + k_p * e * (∂X̂¹/∂X^τ)

    7. Euler step:
           X^{τ+1/n} = X^τ + (1/n) * v_FBFM

    After every n_inner steps (one interleave block):
        - Call env_feedback_fn() to get (obs_new, action_executed)
        - Append action_executed to Y (shift window or update the target)
        - Update obs ← obs_new for the next block's conditioning

    ── Difference from RTC ──────────────────────────────────────────────
    RTC:  Y is fixed throughout inference (set once from A_prev before loop)
    FBFM: Y is updated after each execution step using real env feedback,
          so the guidance target tracks the actual trajectory in real time.

    ── Interface notes ──────────────────────────────────────────────────
    - env_feedback_fn() signature: () -> (obs: np.ndarray (2,), u: float)
      It should execute one action from the current chunk and return the
      resulting observation and the action value used.
    - The rollout harness (rollout_fbfm below) manages the action chunk
      pointer and passes the correct env_feedback_fn closure each cycle.
    - policy.forward(X_tau, obs_t, tau_t) returns v: (B, H, action_dim)
      Use B=1 and squeeze/unsqueeze as needed.
    - To compute ∂X̂¹/∂X^τ, use torch.autograd.grad with retain_graph=True.
    - W is (H,) numpy; convert to torch and reshape to (H, 1) for broadcasting.
    - Clamp τ away from 0 (e.g., max(τ, 1e-4)) to avoid division by zero.

    ── Related code locations ───────────────────────────────────────────
    - policy definition:    model/dit.py  FlowMatchingDiT.forward()
    - vanilla FM inference: model/inference.py  sample_fm()
    - RTC counterpart:      algo/rtc.py   guided_inference_rtc()
    - rollout harness:      algo/fbfm.py  rollout_fbfm()  (below)
    - experiment runner:    experiments/runner.py

    Args:
        policy:          FlowMatchingDiT (eval mode, on `device`)
        obs:             (obs_dim,) initial observation for this cycle
        X_prev:          (H, action_dim) previous chunk, right-padded
        W:               (H,) mask weights
        n_steps:         total denoising steps
        n_inner:         denoising steps per interleave block
        beta:            max guidance weight β
        env_feedback_fn: Callable () -> (obs_new (2,), u_executed float)
                         Called once per interleave block.
                         If None, falls back to RTC-style (no live feedback).
        device:          torch device string

    Returns:
        actions: (H, action_dim) new action chunk (numpy, original scale)
    """
    # ── TODO: implement FBFM guided inference here ─────────────────────
    # Suggested structure:
    #
    #   X = torch.randn(1, H, action_dim).to(device)
    #   dt = 1.0 / n_steps
    #   n_blocks = n_steps // n_inner
    #   Y = torch.from_numpy(X_prev).float().to(device).unsqueeze(0)  # (1,H,1)
    #   W_t = torch.from_numpy(W).float().to(device).unsqueeze(-1)    # (H,1)
    #   obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0) # (1,2)
    #
    #   for block in range(n_blocks):
    #       for inner in range(n_inner):
    #           tau = (block * n_inner + inner) * dt
    #           ... (steps 1-7 above, with autograd for Jacobian)
    #           X = X + dt * v_FBFM
    #
    #       # Execution feedback
    #       if env_feedback_fn is not None:
    #           obs_new, u_exec = env_feedback_fn()
    #           obs_t = torch.from_numpy(obs_new).float().to(device).unsqueeze(0)
    #           # Update Y: shift window and append u_exec
    #           # (implementation detail: how you update Y is part of your design)
    #
    #   return X[0].detach().cpu().numpy()
    raise NotImplementedError(
        "guided_inference_fbfm is not yet implemented. "
        "See the docstring above for the full algorithm and interface notes."
    )


# -----------------------------------------------------------------------
# FBFM rollout harness
# -----------------------------------------------------------------------

def rollout_fbfm(
    policy: FlowMatchingDiT,
    env,                        # MSDEnv instance (already reset)
    ref_seq: np.ndarray,        # (T_total,) reference position sequence
    n_steps: int = 16,          # total FM denoising steps per cycle
    n_inner: int = 4,           # denoising steps per interleave block
    beta: float = 10.0,         # max guidance weight
    action_stats: dict = None,  # {'mean': float, 'std': float} for denorm
    device: str = "cpu",
) -> dict:
    """
    FBFM rollout: fine-grained interleaving of inference and execution.

    Each large cycle runs n_steps denoising steps total, interleaved with
    (n_steps // n_inner) execution steps. The env_feedback_fn closure
    captures the current chunk pointer and feeds back the executed action
    and resulting observation to the inference loop.

    Args:
        policy:       trained FlowMatchingDiT
        env:          MSDEnv (already reset)
        ref_seq:      (T_total,) reference positions
        n_steps:      total denoising steps per cycle
        n_inner:      denoising steps per interleave block
        beta:         max guidance weight
        action_stats: normalization stats
        device:       torch device

    Returns:
        dict with xs_true, xs_obs, actions, times, ref_seq  — all (T_total,)
    """
    T_total = len(ref_seq)
    H = policy.H
    n_blocks = n_steps // n_inner  # execution steps per large cycle

    xs_true  = np.zeros(T_total)
    xs_obs   = np.zeros(T_total)
    actions  = np.zeros(T_total)
    times    = np.zeros(T_total)

    # Current action chunk: (H, action_dim)
    current_chunk = np.zeros((H, 1), dtype=np.float32)
    chunk_ptr = H  # force re-inference on first step

    obs = env.state.copy()  # (2,)

    # Global time step counter
    t_global = 0

    while t_global < T_total:
        # --- Build A_prev from unexecuted tail of current chunk ---
        n_unexecuted = H - chunk_ptr
        X_prev = np.zeros((H, 1), dtype=np.float32)
        if n_unexecuted > 0:
            X_prev[:n_unexecuted] = current_chunk[chunk_ptr:]
        W = np.zeros(H, dtype=np.float32)
        W[:n_unexecuted] = 1.0

        if action_stats is not None:
            X_prev = (X_prev - action_stats["mean"]) / action_stats["std"]

        # --- Build env_feedback_fn closure ---
        # This closure is called once per interleave block inside guided_inference_fbfm.
        # It executes one action from the *current* chunk (before the new chunk is ready),
        # records the result, and returns (obs_new, u_executed).
        #
        # We use a mutable container to share state between the closure and the outer loop.
        feedback_state = {
            "obs": obs.copy(),
            "t_global": t_global,
            "chunk": current_chunk.copy(),
            "ptr": chunk_ptr,
        }
        recorded_steps = []  # list of (xs_true, xs_obs, u, time) per feedback call

        def env_feedback_fn():
            """Execute one action from the current chunk, record result."""
            ptr = feedback_state["ptr"]
            chunk = feedback_state["chunk"]

            # If chunk is exhausted, use zero action (shouldn't happen in normal flow)
            u = float(chunk[ptr, 0]) if ptr < H else 0.0
            feedback_state["ptr"] = min(ptr + 1, H)

            obs_noisy, _, _, info = env.step(u, add_obs_noise=True)
            feedback_state["obs"] = obs_noisy

            recorded_steps.append({
                "xs_true": info["true_state"][0],
                "xs_obs":  obs_noisy[0],
                "u":       u,
                "t":       info["t"],
            })
            return obs_noisy, u

        # --- Run FBFM guided inference (interleaved with execution) ---
        new_chunk_norm = guided_inference_fbfm(
            policy, obs, X_prev, W,
            n_steps=n_steps, n_inner=n_inner, beta=beta,
            env_feedback_fn=env_feedback_fn,
            device=device,
        )  # (H, 1) normalised

        # Denormalise
        if action_stats is not None:
            current_chunk = new_chunk_norm * action_stats["std"] + action_stats["mean"]
        else:
            current_chunk = new_chunk_norm
        chunk_ptr = 0

        # --- Record the steps that were executed during inference ---
        for step in recorded_steps:
            if t_global >= T_total:
                break
            xs_true[t_global]  = step["xs_true"]
            xs_obs[t_global]   = step["xs_obs"]
            actions[t_global]  = step["u"]
            times[t_global]    = step["t"]
            t_global += 1

        obs = feedback_state["obs"]

    return {
        "xs_true":  xs_true,
        "xs_obs":   xs_obs,
        "actions":  actions,
        "times":    times,
        "ref_seq":  ref_seq,
    }
