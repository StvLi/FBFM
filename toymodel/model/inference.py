"""
model/inference.py — Baseline Flow Matching inference (vanilla FM, no guidance)

Euler integration over tau in [0, 1]:
    X^{tau + 1/n} = X^tau + (1/n) * v_theta(X^tau, obs, tau)

Starting from X^0 ~ N(0, I), integrating n_steps gives X^1 (clean token chunk).

Shapes:
    X:   (B, H, token_dim) = (B, 16, 3)
    obs: (B, obs_dim)      = (B, 3)
    tau: (B,)              scalar in [0, 1]
    v:   (B, H, token_dim) = (B, 16, 3)
"""

import torch
import numpy as np
from model.dit import FlowMatchingDiT


@torch.no_grad()
def sample_fm(
    policy: FlowMatchingDiT,
    obs: np.ndarray,          # (obs_dim,) or (B, obs_dim)
    n_steps: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    token_stats: dict = None,
) -> np.ndarray:
    """
    Vanilla Flow Matching inference (no guidance).

    Euler ODE integration:
        X^{tau + dt} = X^tau + dt * v_theta(X^tau, obs, tau)
        tau: 0 → 1,  dt = 1/n_steps

    Args:
        policy:      trained FlowMatchingDiT
        obs:         (obs_dim,) or (B, obs_dim) — conditioning observation
        n_steps:     number of Euler integration steps (default 20)
        device:      torch device string
        token_stats: {'mean': np.array(3,), 'std': np.array(3,)} for denormalization
                     if None, output stays in normalized space

    Returns:
        chunk: (H, token_dim) or (B, H, token_dim) — predicted token chunk
               in original (denormalized) scale if token_stats provided
    """
    policy.eval()

    squeeze = False
    if obs.ndim == 1:
        obs = obs[np.newaxis]  # (1, obs_dim)
        squeeze = True

    B = obs.shape[0]
    H         = policy.H
    token_dim = policy.token_dim

    # obs_t: (B, obs_dim)
    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)

    # X^0 ~ N(0, I),  shape: (B, H, token_dim)
    X = torch.randn(B, H, token_dim, device=device)

    dt = 1.0 / n_steps

    for i in range(n_steps):
        tau = torch.full((B,), i * dt, dtype=torch.float32, device=device)
        v = policy(X, obs_t, tau)  # (B, H, token_dim)
        X = X + dt * v

    # Denormalize with per-dim vector stats
    if token_stats is not None:
        mean_t = torch.tensor(token_stats["mean"], dtype=torch.float32, device=device)
        std_t  = torch.tensor(token_stats["std"],  dtype=torch.float32, device=device)
        X = X * std_t + mean_t

    result = X.cpu().numpy()

    if squeeze:
        result = result[0]  # (H, token_dim)

    return result
