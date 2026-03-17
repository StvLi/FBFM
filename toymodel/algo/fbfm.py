"""
algo/fbfm.py — Feed-Back Flow Matching (FBFM) rollout

FBFM mask policy (W is per-element, shape (H, token_dim)):
    State dims  (col 0-1):  1 ONLY at positions with real observed state.
                             Predictions are ignored — "预测的都不做数".
                             The count of real states grows as feedback arrives.
    Action dims (col 2):    Same as RTC — 1 for non-padded (old-chunk tail), 0 for padding.

FBFM rollout rhythm (single-thread simulation, n_steps=20, n_inner=4):

    Build Y from old chunk's unexecuted tail → (H, token_dim), right-padded
    Trigger: execute Y[0].action → capture obs   (1 execution step)

    guided_inference_fbfm (5 blocks × 4 denoising steps = 20 total):
        Block 0 (τ steps 0-3):   guidance with W_state=[1,0,...], W_action=[1,..,1,0,..]
          → env_feedback_fn() → execute Y[1].action → update obs
          → W_state grows: [1,1,0,...]
        Block 1 (τ steps 4-7):
          → env_feedback_fn() → W_state grows: [1,1,1,0,...]
        Block 2 (τ steps 8-11):
          → env_feedback_fn() → W_state grows: [1,1,1,1,0,...]
        Block 3 (τ steps 12-15):
          → env_feedback_fn() → W_state grows: [1,1,1,1,1,0,...]
        Block 4 (τ steps 16-19):  final output, NO feedback after

    New chunk → chunk_ptr = d = 4 (skip first 4 tokens)
    s_chunk = 1(trigger) + 4(feedback) = 5

    Y stays FIXED throughout inference; only obs and W_state are updated.

Reference: Thoughts.md §实验组, modeling_rtc.py (state_feedback path)

Shapes:
    obs:   (obs_dim,)     = (2,)
    X^τ:   (H, token_dim) = (16, 3)
    Y:     (H, token_dim) = (16, 3)   guidance target (fixed)
    W:     (H, token_dim)             per-element mask (mutable for state dims)
    chunk: (H, token_dim) = (16, 3)
"""

import numpy as np
import torch
from model.dit import FlowMatchingDiT

ACTION_IDX = 2
STATE_DIM  = 2


def build_fbfm_W(
    n_remaining: int,
    n_real_states: int,
    H: int,
    token_dim: int,
) -> np.ndarray:
    """Build per-element mask for FBFM.

    Args:
        n_remaining:   number of non-padded positions (from old chunk tail)
        n_real_states: number of positions with real observed state so far
        H:             sequence length
        token_dim:     token dimension (3)

    Returns:
        W: (H, token_dim)
            W[:n_real_states, :STATE_DIM] = 1   (real state feedback)
            W[:n_remaining,   ACTION_IDX] = 1   (non-padded action, same as RTC)
    """
    W = np.zeros((H, token_dim), dtype=np.float32)
    W[:n_real_states, :STATE_DIM] = 1.0
    W[:n_remaining,   ACTION_IDX] = 1.0
    return W


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _normalize(arr, stats):
    return (arr - stats["mean"]) / stats["std"]

def _denormalize(arr, stats):
    return arr * stats["std"] + stats["mean"]


# -----------------------------------------------------------------------
# Core FBFM guided inference
# -----------------------------------------------------------------------

def guided_inference_fbfm(
    policy: FlowMatchingDiT,
    obs: np.ndarray,            # (obs_dim,) post-trigger observation
    X_prev: np.ndarray,         # (H, token_dim) guidance target Y (normalised)
    W: np.ndarray,              # (H, token_dim) initial per-element mask
    n_steps: int = 20,
    n_inner: int = 4,
    beta: float = 10.0,
    env_feedback_fn=None,       # () -> (obs_new, u_exec)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:                # (H, token_dim) normalised
    """
    FBFM guided inference with real-time feedback.

    W is (H, token_dim) and is MUTATED during inference:
        After each feedback call, state mask gains 1 more position.
        Action mask stays fixed.

    Structure (n_steps=20, n_inner=4 → n_blocks=5):
        for block in range(5):
            for inner in range(4):
                denoising step with current W
            if block < 4:
                env_feedback_fn() → update obs
                W[:, :STATE_DIM] expands by 1 position

    Args:
        policy:          FlowMatchingDiT (eval, on device)
        obs:             (obs_dim,) post-trigger
        X_prev:          (H, token_dim) normalised, right-padded
        W:               (H, token_dim) initial per-element mask
        n_steps:         total denoising steps
        n_inner:         steps per block
        beta:            max guidance weight
        env_feedback_fn: () -> (obs_new, u_exec); called d times
        device:          torch device

    Returns:
        (H, token_dim) normalised numpy
    """
    policy.eval()
    H = policy.H
    token_dim = policy.token_dim
    n_blocks = n_steps // n_inner

    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
    Y     = torch.from_numpy(X_prev.astype(np.float32)).to(device).unsqueeze(0)
    W_t   = torch.from_numpy(W.copy().astype(np.float32)).to(device).unsqueeze(0)  # (1,H,D), mutable

    X = torch.randn(1, H, token_dim, device=device)
    dt = 1.0 / n_steps

    # Track how many real state positions exist for progressive expansion
    n_real_states = int(W[:, 0].sum())  # derive from initial mask

    for block in range(n_blocks):
        for inner in range(n_inner):
            step_idx = block * n_inner + inner
            tau_val = max(step_idx * dt, 1e-4)
            tau_t = torch.full((1,), tau_val, dtype=torch.float32, device=device)

            X = X.detach().requires_grad_(True)
            v = policy(X, obs_t, tau_t)

            one_minus_tau = 1.0 - tau_val
            X_hat1 = X + one_minus_tau * v

            r_sq = one_minus_tau ** 2 / (tau_val ** 2 + one_minus_tau ** 2)
            k_p = min(beta, one_minus_tau / (tau_val * r_sq))

            e = (Y - X_hat1) * W_t

            g = torch.autograd.grad(
                outputs=X_hat1, inputs=X,
                grad_outputs=e.detach(),
                retain_graph=False, create_graph=False,
            )[0]

            v_guided = v.detach() + k_p * g
            X = X.detach() + dt * v_guided

        # Feedback after every block except the last
        if block < n_blocks - 1 and env_feedback_fn is not None:
            obs_new, _u_exec = env_feedback_fn()
            obs_t = torch.from_numpy(obs_new.astype(np.float32)).to(device).unsqueeze(0)

            # Expand state mask by 1 position
            n_real_states += 1
            if n_real_states <= H:
                W_t[0, n_real_states - 1, :STATE_DIM] = 1.0

    return X[0].detach().cpu().numpy()


# -----------------------------------------------------------------------
# FBFM rollout harness
# -----------------------------------------------------------------------

def rollout_fbfm(
    policy: FlowMatchingDiT,
    env,
    ref_seq: np.ndarray,
    n_steps: int = 20,
    n_inner: int = 4,
    beta: float = 10.0,
    token_stats: dict = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """FBFM rollout — see module docstring for rhythm and mask policy."""
    T_total = len(ref_seq)
    H = policy.H
    token_dim = policy.token_dim
    n_blocks = n_steps // n_inner
    d = n_blocks - 1
    s_chunk = 1 + d

    xs_true = np.zeros(T_total)
    xs_obs  = np.zeros(T_total)
    actions = np.zeros(T_total)
    times   = np.zeros(T_total)

    obs = env.state.copy()

    # Bootstrap: unguided
    A_init = np.zeros((H, token_dim), dtype=np.float32)
    W_init = np.zeros((H, token_dim), dtype=np.float32)
    if token_stats is not None:
        A_init = _normalize(A_init, token_stats)

    first_norm = guided_inference_fbfm(
        policy, obs, A_init, W_init,
        n_steps=n_steps, n_inner=n_inner, beta=beta,
        env_feedback_fn=None, device=device,
    )
    current_chunk = _denormalize(first_norm, token_stats) if token_stats else first_norm
    chunk_ptr = 0

    t_global = 0
    while t_global < T_total:
        # Build Y from unexecuted tail
        n_unexecuted = H - chunk_ptr
        Y_raw = np.zeros((H, token_dim), dtype=np.float32)
        if n_unexecuted > 0:
            Y_raw[:n_unexecuted] = current_chunk[chunk_ptr:]

        # Trigger: execute first action
        u_trigger = float(Y_raw[0, ACTION_IDX]) if n_unexecuted > 0 else 0.0
        chunk_ptr += 1

        obs_noisy, _, _, info = env.step(u_trigger, add_obs_noise=True)
        if t_global < T_total:
            xs_true[t_global] = info["true_state"][0]
            xs_obs[t_global]  = obs_noisy[0]
            actions[t_global] = u_trigger
            times[t_global]   = info["t"]
            t_global += 1

        obs_for_inference = obs_noisy.copy()

        # Initial W: 1 real state (trigger), action mask same as RTC
        W = build_fbfm_W(n_unexecuted, n_real_states=1, H=H, token_dim=token_dim)

        if token_stats is not None:
            Y_norm = _normalize(Y_raw, token_stats)
        else:
            Y_norm = Y_raw

        # env_feedback_fn closure
        feedback_state = {"y_ptr": 1, "obs": obs_noisy.copy()}
        recorded_steps = []

        def env_feedback_fn():
            y_ptr = feedback_state["y_ptr"]
            u = float(Y_raw[y_ptr, ACTION_IDX]) if y_ptr < n_unexecuted else 0.0
            feedback_state["y_ptr"] = y_ptr + 1

            obs_new, _, _, info_fb = env.step(u, add_obs_noise=True)
            feedback_state["obs"] = obs_new
            recorded_steps.append({
                "xs_true": info_fb["true_state"][0],
                "xs_obs":  obs_new[0],
                "u":       u,
                "t":       info_fb["t"],
            })
            return obs_new, u

        # Guided inference (W will be mutated inside for state dims)
        new_norm = guided_inference_fbfm(
            policy, obs_for_inference, Y_norm, W,
            n_steps=n_steps, n_inner=n_inner, beta=beta,
            env_feedback_fn=env_feedback_fn, device=device,
        )
        current_chunk = _denormalize(new_norm, token_stats) if token_stats else new_norm

        # Record feedback steps
        for step in recorded_steps:
            if t_global >= T_total:
                break
            xs_true[t_global] = step["xs_true"]
            xs_obs[t_global]  = step["xs_obs"]
            actions[t_global] = step["u"]
            times[t_global]   = step["t"]
            t_global += 1

        chunk_ptr = d
        obs = feedback_state["obs"]

    return {
        "xs_true": xs_true, "xs_obs": xs_obs,
        "actions": actions, "times": times, "ref_seq": ref_seq,
    }
