"""
algo/fm.py — Baseline Flow Matching rollout (no guidance)

Rollout rhythm (every large cycle):
    - Execute s_chunk steps with the current action chunk
    - Re-infer a new action chunk from scratch (vanilla FM, no feedback)
    - Repeat

This is the simplest baseline: the policy is blind to what happened
during execution and simply re-plans every s_chunk steps.

Shapes throughout:
    obs:     (obs_dim,)        = (2,)    — [x, x_dot]
    chunk:   (H, action_dim)   = (16, 1) — action sequence
    action:  (action_dim,)     = (1,)    — single action
"""

import numpy as np
import torch
from model.dit import FlowMatchingDiT
from model.inference import sample_fm


def rollout_fm(
    policy: FlowMatchingDiT,
    env,                        # MSDEnv instance (already reset)
    ref_seq: np.ndarray,        # (T_total,) reference position sequence
    s_chunk: int = 5,           # steps executed per inference cycle
    n_steps: int = 16,          # FM denoising steps
    action_stats: dict = None,  # {'mean': float, 'std': float} for denorm
    device: str = "cpu",
) -> dict:
    """
    Vanilla FM rollout: re-infer every s_chunk steps, no feedback.

    Args:
        policy:       trained FlowMatchingDiT
        env:          MSDEnv (already reset to desired initial state)
        ref_seq:      (T_total,) reference positions
        s_chunk:      execution steps per inference cycle
        n_steps:      FM denoising steps
        action_stats: normalization stats for denormalization
        device:       torch device

    Returns:
        dict with:
            xs_true:   (T_total,)  true position trajectory
            xs_obs:    (T_total,)  observed (noisy) position
            actions:   (T_total,)  executed actions
            times:     (T_total,)  time stamps
    """
    # 这里的T_total不是FM去噪步数n_steps(通常比如16)，而是整个仿真轨迹的总时间步数，
    # 由参考轨迹ref_seq长度决定，通常几百步。n_steps只是生成一段动作块时FM内部的去噪迭代步数。
    T_total = len(ref_seq)

    # 初始化轨迹、观测、动作和时间
    xs_true  = np.zeros(T_total)  # 真实轨迹
    xs_obs   = np.zeros(T_total)  # 观测轨迹
    actions  = np.zeros(T_total)  # 执行动作
    times    = np.zeros(T_total)  # 时间戳

    # 当前动作块: (H, action_dim) = (16, 1)
    # 初始化为零; 将在第一次执行步骤前被替换
    H = policy.H # 动作块长度，通常为16
    current_chunk = np.zeros((H, 1), dtype=np.float32)

    # 关于 chunk_ptr=H 的时序机制说明：
    # 在本 FM rollout 循环实现中，每当 chunk_ptr >= s_chunk 时（首次循环等同于 chunk_ptr=H），
    # 会立即（在本时刻，不做等待）重新推理生成一个新的动作块 current_chunk（调用 sample_fm），
    # 并立刻把 chunk_ptr 重置为 0，然后本次循环就会从新块的第0个动作开始执行。
    # 换句话说，只要 chunk 指针已经指向块末（chunk_ptr >= s_chunk），
    # 就会发生一次推理，下一步立刻取用新块的第一个动作。
    # 这种实现确保每经过 s_chunk 步就同步推理一次，
    # 没有额外延迟，也不会出现 chunk 之间的观测空窗期。
    chunk_ptr = H  # 初始化为H，确保主循环第一步立刻推理并从新块第0步开始执行

    # 当前观测 (从 env.state 获取, 开始时没有噪声)
    obs = env.state.copy()  # (2,)

    for t in range(T_total):
        # --- 推理: 当动作块耗尽时重新规划 ---
        if chunk_ptr >= s_chunk:
            # 当本轮动作序列（动作块）已经全部执行完毕，会发生以下几件事：
            # 1. 立刻用当前观测 obs 重新推理，生成下一个全新的动作序列（动作块）
            # 2. 将 chunk_ptr 归零，指向新动作块的第 0 步
            # 3. 接下来的环境步进将会从新推理出的动作块第一个动作重新执行
            # 这样保证每经过 s_chunk 步就会完整 replan，一直循环，覆盖整个仿真时间轴
            current_chunk = sample_fm(
                policy, obs, n_steps=n_steps,
                device=device, action_stats=action_stats,
            )  # (H, 1) 原始尺度
            chunk_ptr = 0

        # --- 执行: 从动作块中取下一个动作 ---
        u = float(current_chunk[chunk_ptr, 0])
        chunk_ptr += 1

        # 环境步进
        obs_noisy, _, _, info = env.step(u, add_obs_noise=True)

        # 记录
        xs_true[t]  = info["true_state"][0]
        xs_obs[t]   = obs_noisy[0]
        actions[t]  = u
        times[t]    = info["t"]

        # 更新观测用于下次推理 (使用噪声观测, 就像真实部署一样)
        obs = obs_noisy

    return {
        "xs_true":  xs_true,   # (T_total,)
        "xs_obs":   xs_obs,    # (T_total,)
        "actions":  actions,   # (T_total,)
        "times":    times,     # (T_total,)
        "ref_seq":  ref_seq,   # (T_total,)
    }
