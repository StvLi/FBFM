"""
algo/rtc.py — Real-Time Chunking (RTC) rollout

RTC rollout rhythm (single-thread simulation of async dual-thread):
    Large cycle:
        [Execution chain]  Execute s_chunk steps (first step reads obs as FM input,
                           remaining s_chunk-1 steps execute blindly)
        [Inference chain]  Full guided FM inference (n_denoising steps)
        → next large cycle

Key idea: the flow field is updated by gradient feedback from the previous
chunk's unexecuted actions (A_prev), making transitions smoother. But during
execution, the controller is blind — if the environment changes mid-chunk,
there is no within-chunk correction.

Reference: RTC Paper.pdf, Thoughts.md §对照二

Shapes throughout:
    obs:     (obs_dim,)          = (2,)     — [x, x_dot]
    A_prev:  (H, action_dim)     = (16, 1)  — previous chunk (right-padded with zeros)
    W:       (H,)                           — mask weights (1 for executed, 0 for padded)
    chunk:   (H, action_dim)     = (16, 1)  — new action sequence
"""

import numpy as np
import torch
from model.dit import FlowMatchingDiT


# -----------------------------------------------------------------------
# [USER IMPLEMENTS] Core RTC guided inference
# -----------------------------------------------------------------------

def guided_inference_rtc(
    policy: FlowMatchingDiT,
    obs: np.ndarray,            # (obs_dim,) = (2,)  current observation
    A_prev: np.ndarray,         # (H, action_dim) = (16, 1)  previous unexecuted actions,
                                #   right-padded with zeros to length H
    W: np.ndarray,              # (H,)  mask weights: 1.0 for real entries, 0.0 for padding
    n_steps: int = 16,          # total denoising steps
    beta: float = 10.0,         # max guidance weight
    device: str = "cpu",
) -> np.ndarray:                # (H, action_dim) = (16, 1)  new action chunk
    """
    RTC guided Flow Matching inference.

    This function implements the core RTC update rule. The flow field is
    steered at each denoising step so that the output chunk is consistent
    with the previous chunk's unexecuted actions.

    ── Algorithm (one denoising step at noise level τ) ──────────────────
    Given:
        X^τ  : (H, action_dim)  current noisy sample
        obs  : (obs_dim,)       conditioning observation
        τ    : scalar           current noise level ∈ [0, 1]
        Y    : (H, action_dim)  target = A_prev (right-padded, masked by W)

    1. Predict velocity:
           v = policy(X^τ, obs, τ)          # (H, action_dim)

    2. Predict clean sample:
           X̂¹ = X^τ + (1 - τ) * v           # (H, action_dim)

    3. Compute guidance gain:
           r²_τ = (1-τ)² / (τ² + (1-τ)²)
           k_p  = min(β, (1-τ) / (τ * r²_τ))   # scalar, clamp τ away from 0

    4. Compute guidance gradient (Jacobian of X̂¹ w.r.t. X^τ):
           ∂X̂¹/∂X^τ ≈ I + (1-τ) * ∂v/∂X^τ    # requires torch.autograd

    5. Compute error weighted by mask:
           e = (Y - X̂¹)^T diag(W)             # (H, action_dim), masked

    6. Guided velocity:
           v_guided = v + k_p * e * (∂X̂¹/∂X^τ)

    7. Euler step:
           X^{τ+1/n} = X^τ + (1/n) * v_guided

    ── Interface notes ──────────────────────────────────────────────────
    - policy.forward(X_tau, obs_t, tau_t) returns v: (B, H, action_dim)
      Use B=1 and squeeze/unsqueeze as needed.
    - To compute ∂X̂¹/∂X^τ, use torch.autograd.grad or enable requires_grad
      on X^τ before the forward pass.
    - W is a (H,) numpy array; convert to torch and reshape to (H, 1) for
      broadcasting with (H, action_dim).
    - Clamp τ away from 0 (e.g., max(τ, 1e-4)) to avoid division by zero
      in k_p.

    ── Related code locations ───────────────────────────────────────────
    - policy definition:    model/dit.py  FlowMatchingDiT.forward()
    - vanilla FM inference: model/inference.py  sample_fm()
    - FBFM counterpart:     algo/fbfm.py  guided_inference_fbfm()
    - rollout harness:      algo/rtc.py   rollout_rtc()  (below)
    - experiment runner:    experiments/runner.py

    Args:
        policy:   FlowMatchingDiT (eval mode, on `device`)
        obs:      (obs_dim,) current observation
        A_prev:   (H, action_dim) previous chunk, right-padded with zeros
        W:        (H,) mask weights
        n_steps:  total denoising steps
        beta:     max guidance weight β
        device:   torch device string

    Returns:
        actions: (H, action_dim) new action chunk (numpy, original scale)
    """
    # ── TODO: implement RTC guided inference here ──────────────────────
    # Hint: structure your loop as:
    #   X = torch.randn(1, H, action_dim).to(device)
    #   dt = 1.0 / n_steps
    #   for i in range(n_steps):
    #       tau = i * dt
    #       ... (steps 1-7 above)
    #       X = X + dt * v_guided
    #   return X[0].detach().cpu().numpy()
    raise NotImplementedError(
        "guided_inference_rtc is not yet implemented. "
        "See the docstring above for the full algorithm and interface notes."
    )


# -----------------------------------------------------------------------
# RTC rollout harness (calls guided_inference_rtc each cycle)
# -----------------------------------------------------------------------

def rollout_rtc(
    policy: FlowMatchingDiT,
    env,                        # MSDEnv instance (already reset)
    ref_seq: np.ndarray,        # (T_total,) reference position sequence
    s_chunk: int = 5,           # execution steps per large cycle
    n_steps: int = 16,          # FM denoising steps
    beta: float = 10.0,         # max guidance weight
    action_stats: dict = None,  # {'mean': float, 'std': float} for denorm
    device: str = "cpu",
) -> dict:
    """
    RTC rollout: guided FM inference every s_chunk steps.

    Single-thread simulation of the async dual-thread RTC rhythm:
        1. Read obs at start of cycle
        2. Execute s_chunk actions from current chunk (blindly)
        3. Run guided_inference_rtc with A_prev = unexecuted tail of old chunk
        4. Replace current chunk with new chunk
        5. Repeat

    Args:
        policy:       trained FlowMatchingDiT
        env:          MSDEnv (already reset)
        ref_seq:      (T_total,) reference positions
        s_chunk:      execution steps per cycle
        n_steps:      FM denoising steps
        beta:         max guidance weight
        action_stats: normalization stats (applied inside guided_inference_rtc)
        device:       torch device

    Returns:
        dict with xs_true, xs_obs, actions, times, ref_seq  — all (T_total,)
    """
    T_total = len(ref_seq)
    H = policy.H

    xs_true  = np.zeros(T_total)
    xs_obs   = np.zeros(T_total)
    actions  = np.zeros(T_total)
    times    = np.zeros(T_total)

    # Current action chunk: (H, action_dim)
    current_chunk = np.zeros((H, 1), dtype=np.float32)
    chunk_ptr = H  # force re-inference on first step

    obs = env.state.copy()  # (2,)

    for t in range(T_total):
        # --- Inference trigger ---
        if chunk_ptr >= s_chunk:
            # Build A_prev: unexecuted tail of current chunk, right-padded to H
            # A_prev: (H, action_dim)
            n_unexecuted = H - chunk_ptr
            A_prev = np.zeros((H, 1), dtype=np.float32)
            if n_unexecuted > 0:
                A_prev[:n_unexecuted] = current_chunk[chunk_ptr:]
            # W: (H,) — 1.0 for real entries, 0.0 for zero-padded entries
            W = np.zeros(H, dtype=np.float32)
            W[:n_unexecuted] = 1.0

            # Normalise A_prev if stats provided
            if action_stats is not None:
                A_prev = (A_prev - action_stats["mean"]) / action_stats["std"]

            # Guided inference
            new_chunk_norm = guided_inference_rtc(
                policy, obs, A_prev, W,
                n_steps=n_steps, beta=beta, device=device,
            )  # (H, 1) in normalised space

            # Denormalise
            if action_stats is not None:
                current_chunk = new_chunk_norm * action_stats["std"] + action_stats["mean"]
            else:
                current_chunk = new_chunk_norm

            chunk_ptr = 0

        # --- Execution ---
        u = float(current_chunk[chunk_ptr, 0])
        chunk_ptr += 1

        obs_noisy, _, _, info = env.step(u, add_obs_noise=True)

        xs_true[t]  = info["true_state"][0]
        xs_obs[t]   = obs_noisy[0]
        actions[t]  = u
        times[t]    = info["t"]

        obs = obs_noisy

    return {
        "xs_true":  xs_true,
        "xs_obs":   xs_obs,
        "actions":  actions,
        "times":    times,
        "ref_seq":  ref_seq,
    }
