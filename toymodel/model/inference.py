"""
model/inference.py — Baseline Flow Matching inference (vanilla FM, no guidance)

Euler integration over tau in [0, 1]:
    X^{tau + 1/n} = X^tau + (1/n) * v_theta(X^tau, obs, tau)

Starting from X^0 ~ N(0, I), integrating n_steps gives X^1 (clean token chunk).

Supports inpainting: when prev_chunk_norm and d>0 are provided, the first d
positions are forced to the old chunk's values at each step, with tau=1.0
(clean). This ensures the new chunk's prefix matches the old chunk exactly,
so action continuity is preserved across chunk boundaries.

Shapes:
    X:   (B, H, token_dim) = (B, 16, 3)
    obs: (B, obs_dim)      = (B, 3)
    tau: (B, H)            per-position in [0, 1]
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
    prev_chunk_norm: np.ndarray = None,  # (H, token_dim) normalised old chunk for inpainting
    d: int = 0,                          # inference delay — first d positions are inpainted
) -> np.ndarray:
    """
    Flow Matching inference with optional inpainting.

    Euler ODE integration:
        X^{tau + dt} = X^tau + dt * v_theta(X^tau, obs, tau)
        tau: 0 → 1,  dt = 1/n_steps

    Inpainting (when prev_chunk_norm is not None and d > 0):
        At each denoising step, the first d positions of X are replaced with
        the corresponding positions from prev_chunk_norm (at their current
        interpolation level). Per-position tau is set to 1.0 for frozen
        positions, so the model sees them as clean data.

    Args:
        policy:          trained FlowMatchingDiT
        obs:             (obs_dim,) or (B, obs_dim) — conditioning observation
        n_steps:         number of Euler integration steps (default 20)
        device:          torch device string
        token_stats:     {'mean': np.array(3,), 'std': np.array(3,)} for denormalization
        prev_chunk_norm: (H, token_dim) normalised old chunk for inpainting prefix
        d:               number of frozen prefix positions (inference delay)

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

    # Prepare inpainting target if provided
    do_inpaint = prev_chunk_norm is not None and d > 0
    if do_inpaint:
        prev_t = torch.from_numpy(
            prev_chunk_norm.astype(np.float32)
        ).to(device).unsqueeze(0).expand(B, -1, -1)  # (B, H, token_dim)

    dt = 1.0 / n_steps

    for i in range(n_steps):
        tau_val = i * dt

        # Inpainting: force first d positions toward old chunk
        if do_inpaint:
            # At tau_val, the expected clean-data interpolation is:
            #   X^tau = tau * X1 + (1-tau) * X0
            # For frozen positions we want X = prev_t (the clean target),
            # so we set them to prev_t directly (they'll have tau=1.0).
            X[:, :d, :] = prev_t[:, :d, :]

        # Per-position tau: (B, H)
        tau_pp = torch.full((B, H), tau_val, dtype=torch.float32, device=device)
        if do_inpaint:
            tau_pp[:, :d] = 1.0  # frozen positions are clean

        v = policy(X, obs_t, tau_pp)  # (B, H, token_dim)
        X = X + dt * v

    # Final inpainting clamp
    if do_inpaint:
        X[:, :d, :] = prev_t[:, :d, :]

    # Denormalize with per-dim vector stats
    if token_stats is not None:
        mean_t = torch.tensor(token_stats["mean"], dtype=torch.float32, device=device)
        std_t  = torch.tensor(token_stats["std"],  dtype=torch.float32, device=device)
        X = X * std_t + mean_t

    result = X.cpu().numpy()

    if squeeze:
        result = result[0]  # (H, token_dim)

    return result
