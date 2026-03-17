"""
model/inference.py — Baseline Flow Matching inference (vanilla FM, no guidance)

Euler integration over tau in [0, 1]:
    X^{tau + 1/n} = X^tau + (1/n) * v_theta(X^tau, obs, tau)

Starting from X^0 ~ N(0, I), integrating n_steps gives X^1 (clean action chunk).

Shapes:
    X:   (B, H, action_dim) = (B, 16, 1)
    obs: (B, obs_dim)       = (B, 2)
    tau: (B,)               scalar in [0, 1]
    v:   (B, H, action_dim) = (B, 16, 1)
"""

import torch
import numpy as np
from model.dit import FlowMatchingDiT


@torch.no_grad()
def sample_fm(
    policy: FlowMatchingDiT,
    obs: np.ndarray,          # (obs_dim,) or (B, obs_dim)
    n_steps: int = 16,
    device: str = "cpu",
    action_stats: dict = None,
) -> np.ndarray:
    """
    Vanilla Flow Matching inference (no guidance).

    Euler ODE integration:
        X^{tau + dt} = X^tau + dt * v_theta(X^tau, obs, tau)
        tau: 0 → 1,  dt = 1/n_steps

    Args:
        policy:       trained FlowMatchingDiT
        obs:          (obs_dim,) or (B, obs_dim) — conditioning observation
        n_steps:      number of Euler integration steps
        device:       torch device string
        action_stats: {'mean': float, 'std': float} for denormalization
                      if None, output stays in normalized space

    Returns:
        actions: (H, action_dim) or (B, H, action_dim) — predicted action chunk
                 in original (denormalized) scale if action_stats provided
    """
    policy.eval()

    # Handle single obs vs batch
    squeeze = False
    if obs.ndim == 1:
        obs = obs[np.newaxis]  # (1, obs_dim)
        squeeze = True

    B = obs.shape[0]
    H          = policy.H
    action_dim = policy.action_dim

    # obs_t: (B, obs_dim)
    obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)

    # X^0 ~ N(0, I),  shape: (B, H, action_dim)
    X = torch.randn(B, H, action_dim, device=device)

    dt = 1.0 / n_steps

    for i in range(n_steps):
        # tau: (B,)  — current noise level
        tau = torch.full((B,), i * dt, dtype=torch.float32, device=device)

        # v: (B, H, action_dim)
        v = policy(X, obs_t, tau)

        # Euler step
        X = X + dt * v

    # Denormalize if stats provided
    if action_stats is not None:
        X = X * action_stats["std"] + action_stats["mean"]

    actions = X.cpu().numpy()

    if squeeze:
        actions = actions[0]  # (H, action_dim)

    return actions
