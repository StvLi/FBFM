"""
algo/fm.py — Baseline Flow Matching rollout (no guidance)

Rollout rhythm (with simulated inference delay, same as RTC/FBFM):
    Bootstrap: infer first chunk from initial obs (no delay)
    Each large cycle:
        1. Trigger: execute chunk[chunk_ptr], capture obs_for_inference
        2. Blind execution: execute d more steps from current chunk
        3. Inference: sample_fm(obs_for_inference) -> new chunk (no guidance)
        4. chunk_ptr = d (skip first d tokens — they correspond to blind period)

Shapes throughout:
    obs:     (obs_dim,)        = (3,)    — [x, x_dot, x_ref]
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
    Vanilla FM rollout with simulated inference delay (same timing as RTC/FBFM).

    Rhythm: trigger(1) + blind(d) + inference -> new chunk from chunk[d]
    """
    T_total = len(ref_seq)
    H = policy.H
    token_dim = policy.token_dim
    d = s_chunk - 1  # inference delay = blind execution steps

    xs_true = np.zeros(T_total)
    xs_obs  = np.zeros(T_total)
    actions = np.zeros(T_total)
    times   = np.zeros(T_total)

    obs = np.append(env.state.copy(), ref_seq[0])  # (3,)

    # Bootstrap: first chunk, no delay
    current_chunk = sample_fm(
        policy, obs, n_steps=n_steps,
        device=device, token_stats=token_stats,
    )  # (H, token_dim)
    chunk_ptr = 0

    t_global = 0
    while t_global < T_total:
        # Trigger: execute one action, capture obs for next inference
        u = float(current_chunk[chunk_ptr, ACTION_IDX]) if chunk_ptr < H else 0.0
        chunk_ptr += 1
        obs_noisy, _, _, info = env.step(u, add_obs_noise=True)
        xs_true[t_global] = info["true_state"][0]
        xs_obs[t_global]  = obs_noisy[0]
        actions[t_global] = u
        times[t_global]   = info["t"]
        t_global += 1
        obs_for_inference = np.append(obs_noisy, ref_seq[min(t_global, T_total - 1)])

        # Blind execution: d steps using current chunk
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

        # Inference: vanilla FM, no guidance (uses trigger-time obs)
        current_chunk = sample_fm(
            policy, obs_for_inference, n_steps=n_steps,
            device=device, token_stats=token_stats,
        )
        chunk_ptr = d  # skip first d tokens (correspond to blind period)

    return {
        "xs_true": xs_true, "xs_obs": xs_obs,
        "actions": actions, "times": times, "ref_seq": ref_seq,
    }
