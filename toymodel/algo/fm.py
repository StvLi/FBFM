"""
algo/fm.py — Baseline Flow Matching rollout (no guidance)

Rollout rhythm (every large cycle):
    - Execute s_chunk steps with the current token chunk
    - Re-infer a new token chunk from scratch (vanilla FM, no feedback)
    - Repeat

This is the simplest baseline: the policy is blind to what happened
during execution and simply re-plans every s_chunk steps.

Shapes throughout:
    obs:     (obs_dim,)        = (2,)    — [x, x_dot]
    chunk:   (H, token_dim)    = (16, 3) — token sequence [x, x_dot, u]
    action:  scalar                      — u extracted from chunk[:, 2]
"""

import numpy as np
import torch
from model.dit import FlowMatchingDiT
from model.inference import sample_fm

ACTION_IDX = 2  # index of action dimension within the token


def rollout_fm(
    policy: FlowMatchingDiT,
    env,                        # MSDEnv instance (already reset)
    ref_seq: np.ndarray,        # (T_total,) reference position sequence
    s_chunk: int = 5,           # steps executed per inference cycle
    n_steps: int = 20,          # FM denoising steps
    token_stats: dict = None,   # {'mean': np.array(3,), 'std': np.array(3,)} for denorm
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Vanilla FM rollout: re-infer every s_chunk steps, no feedback.

    Args:
        policy:       trained FlowMatchingDiT
        env:          MSDEnv (already reset to desired initial state)
        ref_seq:      (T_total,) reference positions
        s_chunk:      execution steps per inference cycle
        n_steps:      FM denoising steps (default 20)
        token_stats:  per-dim normalization stats for denormalization
        device:       torch device

    Returns:
        dict with:
            xs_true:   (T_total,)  true position trajectory
            xs_obs:    (T_total,)  observed (noisy) position
            actions:   (T_total,)  executed actions
            times:     (T_total,)  time stamps
    """
    T_total = len(ref_seq)

    xs_true  = np.zeros(T_total)
    xs_obs   = np.zeros(T_total)
    actions  = np.zeros(T_total)
    times    = np.zeros(T_total)

    H = policy.H
    token_dim = policy.token_dim
    # 当前 token 块: (H, token_dim) = (16, 3)
    current_chunk = np.zeros((H, token_dim), dtype=np.float32)

    # chunk_ptr=H 确保主循环第一步立刻推理
    chunk_ptr = H

    obs = env.state.copy()  # (2,)

    for t in range(T_total):
        if chunk_ptr >= s_chunk:
            current_chunk = sample_fm(
                policy, obs, n_steps=n_steps,
                device=device, token_stats=token_stats,
            )  # (H, token_dim) 原始尺度
            chunk_ptr = 0

        # 提取 action = token 的第 2 列 (u)
        u = float(current_chunk[chunk_ptr, ACTION_IDX])
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
