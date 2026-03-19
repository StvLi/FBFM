"""
algo/rtc.py — Real-Time Chunking (RTC) rollout

RTC rollout rhythm (single-thread simulation of async dual-thread):
    Large cycle:
        [Execution chain]  Execute s_chunk steps (first step reads obs as FM input,
                           remaining s_chunk-1 steps execute blindly)
        [Inference chain]  Full guided FM inference (n_denoising steps)
        → next large cycle

Key idea: the flow field is updated by gradient feedback from the previous
chunk's unexecuted tokens (A_prev), making transitions smoother. But during
execution, the controller is blind — if the environment changes mid-chunk,
there is no within-chunk correction.

Reference: RTC Paper.pdf, Thoughts.md §对照二

Shapes throughout:
    obs:     (obs_dim,)          = (3,)     — [x, x_dot, x_ref]
    A_prev:  (H, token_dim)      = (16, 3)  — previous chunk (right-padded with zeros)
    W:       (H, token_dim)                 — per-element mask (see build_rtc_W)
    chunk:   (H, token_dim)      = (16, 3)  — token sequence [x, x_dot, u]

RTC mask policy:
    W[:, 0:2] (state dims)  = 0  always — RTC does NOT use state for guidance
    W[:, 2]   (action dim)  = 1  for non-padded positions, 0 for zero-padding
"""

import numpy as np
import torch
from model.dit import FlowMatchingDiT

ACTION_IDX = 2
STATE_DIM  = 2


def build_rtc_W(n_remaining: int, H: int, token_dim: int,
                d: int = 0, s: int = 5) -> np.ndarray:
    """Build per-element soft mask for RTC (Paper Eq.5).

    Three-region mask on the action dimension:
        W[i<d]       = 1          (frozen, will be executed during inference)
        W[d<=i<H-s]  = exp decay  (intermediate, can be updated)
        W[i>=H-s]    = 0          (beyond old chunk, freshly generated)

    State dimensions are always 0 (RTC does not use state guidance).

    Args:
        n_remaining: non-padded positions from old chunk tail
        H:           sequence length
        token_dim:   token dimension (3)
        d:           inference delay (frozen prefix length)
        s:           execution horizon

    Returns:
        W: (H, token_dim)
    """
    W = np.zeros((H, token_dim), dtype=np.float32)
    for i in range(min(n_remaining, H)):
        if i < d:
            w = 1.0
        elif i < H - s:
            denom = H - s - d + 1
            if denom > 0:
                ci = (H - s - i) / denom
                w = ci * (np.exp(ci) - 1) / (np.e - 1)
            else:
                w = 0.0
        else:
            w = 0.0
        W[i, ACTION_IDX] = w
    return W


# -----------------------------------------------------------------------
# Core RTC guided inference
# -----------------------------------------------------------------------

def guided_inference_rtc(
    policy: FlowMatchingDiT,
    obs: np.ndarray,            # (obs_dim,) = (3,) = [x, x_dot, x_ref]
    A_prev: np.ndarray,         # (H, token_dim) = (16, 3)  normalised, right-padded
    W: np.ndarray,              # (H, token_dim) per-element mask
    n_steps: int = 20,
    beta: float = 10.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:                # (H, token_dim)  normalised
    """
    RTC guided Flow Matching inference.

    W is now (H, token_dim) per-element mask — for RTC only the action column
    is non-zero (state columns are always 0).

    Algorithm per denoising step τ:
        v = policy(X^τ, obs, τ)
        X̂¹ = X^τ + (1−τ)·v
        e = (Y − X̂¹) * W             ← per-element mask
        g = eᵀ · ∂X̂¹/∂X^τ            ← VJP via autograd
        v_guided = v + k_p · g
        X^{τ+dt} = X^τ + dt · v_guided

    Args:
        policy:  FlowMatchingDiT (eval, on device)
        obs:     (obs_dim,)
        A_prev:  (H, token_dim) normalised
        W:       (H, token_dim) per-element mask
        n_steps: denoising steps (default 20)
        beta:    max guidance weight
        device:  torch device

    Returns:
        (H, token_dim) normalised numpy
    """
    policy.eval()
    H = policy.H
    token_dim = policy.token_dim

    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
    Y     = torch.from_numpy(A_prev.astype(np.float32)).to(device).unsqueeze(0)   # (1,H,D)
    W_t   = torch.from_numpy(W.astype(np.float32)).to(device).unsqueeze(0)        # (1,H,D)

    X = torch.randn(1, H, token_dim, device=device)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        tau_val = max(i * dt, 1e-4)
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

    return X[0].detach().cpu().numpy()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _normalize(arr, stats):
    return (arr - stats["mean"]) / stats["std"]

def _denormalize(arr, stats):
    return arr * stats["std"] + stats["mean"]


# -----------------------------------------------------------------------
# RTC rollout
# -----------------------------------------------------------------------

def rollout_rtc(
    policy: FlowMatchingDiT,
    env,
    ref_seq: np.ndarray,
    s_chunk: int = 5,
    n_steps: int = 20,
    beta: float = 10.0,
    token_stats: dict = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """RTC rollout — see module docstring for rhythm details."""
    T_total = len(ref_seq)
    H = policy.H
    token_dim = policy.token_dim
    d = s_chunk - 1

    xs_true = np.zeros(T_total)
    xs_obs  = np.zeros(T_total)
    actions = np.zeros(T_total)
    times   = np.zeros(T_total)

    obs = np.append(env.state.copy(), ref_seq[0])  # (3,) = [x, x_dot, x_ref]

    # Bootstrap
    A_prev_init = np.zeros((H, token_dim), dtype=np.float32)
    W_init = np.zeros((H, token_dim), dtype=np.float32)
    if token_stats is not None:
        A_prev_init = _normalize(A_prev_init, token_stats)

    first_norm = guided_inference_rtc(
        policy, obs, A_prev_init, W_init,
        n_steps=n_steps, beta=beta, device=device,
    )
    current_chunk = _denormalize(first_norm, token_stats) if token_stats else first_norm
    chunk_ptr = 0

    t_global = 0
    while t_global < T_total:
        # Trigger
        u = float(current_chunk[chunk_ptr, ACTION_IDX]) if chunk_ptr < H else 0.0
        chunk_ptr += 1
        obs_noisy, _, _, info = env.step(u, add_obs_noise=True)
        xs_true[t_global] = info["true_state"][0]
        xs_obs[t_global]  = obs_noisy[0]
        actions[t_global] = u
        times[t_global]   = info["t"]
        t_global += 1
        obs_for_inference = np.append(obs_noisy, ref_seq[min(t_global, T_total - 1)])  # (3,)

        # Build A_prev + W
        n_remaining = H - chunk_ptr
        A_prev = np.zeros((H, token_dim), dtype=np.float32)
        if n_remaining > 0:
            A_prev[:n_remaining] = current_chunk[chunk_ptr:]
        W = build_rtc_W(n_remaining, H, token_dim, d=d, s=s_chunk)
        if token_stats is not None:
            A_prev = _normalize(A_prev, token_stats)

        # Blind execution
        for _ in range(d):
            if t_global >= T_total:
                break
            u = float(current_chunk[chunk_ptr, ACTION_IDX]) if chunk_ptr < H else 0.0
            chunk_ptr += 1
            obs_noisy, _, _, info = env.step(u, add_obs_noise=True)
            xs_true[t_global] = info["true_state"][0]
            xs_obs[t_global]  = obs_noisy[0]
            actions[t_global] = u
            times[t_global]   = info["t"]
            t_global += 1

        # Guided inference
        new_norm = guided_inference_rtc(
            policy, obs_for_inference, A_prev, W,
            n_steps=n_steps, beta=beta, device=device,
        )
        current_chunk = _denormalize(new_norm, token_stats) if token_stats else new_norm
        chunk_ptr = d

    return {
        "xs_true": xs_true, "xs_obs": xs_obs,
        "actions": actions, "times": times, "ref_seq": ref_seq,
    }
