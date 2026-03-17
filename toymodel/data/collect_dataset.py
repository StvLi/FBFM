"""
data/collect_dataset.py — PID 专家数据集采集

采集流程：
    1. 随机生成参考轨迹（阶跃 or 正弦）
    2. PID 控制器跟踪，记录 (state, action) 对
    3. 将轨迹切分为长度 H 的 chunk，存为 numpy 数组

输出数据格式（保存到 data/expert_dataset.npz）：
    states:   (N, H+1, state_dim)  — 每条轨迹的状态序列（含初始状态）
              state_dim = 2: [x, x_dot]
    actions:  (N, H, action_dim)   — 每条轨迹的动作序列
              action_dim = 1: [u]
    refs:     (N, H, 1)            — 对应的参考位置序列

其中 N = n_trajs * chunks_per_traj
"""

import sys
import os
import numpy as np

# 将项目根目录加入 path，方便 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.msd_env import MSDEnv
from sim.pid_controller import PIDController


# -----------------------------------------------------------------------
# 参考轨迹生成
# -----------------------------------------------------------------------

def make_step_ref(t_arr: np.ndarray, amp: float = 1.0) -> np.ndarray:
    """
    阶跃参考轨迹：t >= 0 时跳变到 amp。

    Args:
        t_arr: (T,) 时间数组
        amp:   阶跃幅值 (m)

    Returns:
        ref: (T,) 参考位置
    """
    return np.full_like(t_arr, amp)


def make_sin_ref(
    t_arr: np.ndarray,
    amp: float = 1.0,
    freq: float = 0.5,
    phase: float = 0.0,
) -> np.ndarray:
    """
    正弦参考轨迹：x_ref(t) = amp * sin(2π * freq * t + phase)

    Args:
        t_arr: (T,) 时间数组
        amp:   幅值 (m)
        freq:  频率 (Hz)
        phase: 初相 (rad)

    Returns:
        ref: (T,) 参考位置
    """
    return amp * np.sin(2 * np.pi * freq * t_arr + phase)


# -----------------------------------------------------------------------
# 单条轨迹采集
# -----------------------------------------------------------------------

def collect_one_trajectory(
    env: MSDEnv,
    pid: PIDController,
    ref_seq: np.ndarray,   # (T,) 参考位置序列
    add_obs_noise: bool = True,
) -> dict:
    """
    用 PID 控制器跟踪 ref_seq，采集一条完整轨迹。
    内部会调用 env.reset()（从默认初始状态出发）。
    如需自定义初始状态，请在调用前手动 reset，然后用 collect_one_trajectory_no_reset。

    Args:
        env:           MSDEnv 实例
        pid:           PIDController 实例（已 reset）
        ref_seq:       (T,) 参考位置序列，长度决定轨迹步数
        add_obs_noise: 是否在观测上加噪声

    Returns:
        dict with:
            states:  (T+1, 2)  — 状态序列（含初始状态）
            actions: (T, 1)    — 动作序列
            refs:    (T, 1)    — 参考位置序列
            t_arr:   (T,)      — 时间序列
    """
    env.reset()
    return collect_one_trajectory_no_reset(env, pid, ref_seq, add_obs_noise)


def collect_one_trajectory_no_reset(
    env: MSDEnv,
    pid: PIDController,
    ref_seq: np.ndarray,   # (T,) 参考位置序列
    add_obs_noise: bool = True,
) -> dict:
    """
    采集一条轨迹，不在内部调用 reset（由调用方负责）。
    用于 collect_dataset 中需要自定义初始状态的场景。
    """
    T = len(ref_seq)

    # state: (2,) — 读取当前环境状态作为初始状态
    init_state = env.state.copy()

    # 预分配
    states  = np.zeros((T + 1, 2),  dtype=np.float32)  # (T+1, state_dim)
    actions = np.zeros((T,    1),   dtype=np.float32)  # (T, action_dim)
    refs    = np.zeros((T,    1),   dtype=np.float32)  # (T, 1)
    t_arr   = np.zeros(T,           dtype=np.float32)  # (T,)

    states[0] = init_state  # 初始状态（无噪声）

    for i in range(T):
        x_ref = float(ref_seq[i])

        # PID 计算控制力
        # 注意：PID 用带噪声的观测（模拟真实控制器）
        u = pid.compute(states[i], x_ref)

        # 环境执行一步
        obs, _, _, info = env.step(u, add_obs_noise=add_obs_noise)

        # 记录
        actions[i]     = u                          # (1,)
        refs[i]        = x_ref                      # (1,)
        t_arr[i]       = info["t"]
        states[i + 1]  = info["true_state"]         # 存真实状态（无噪声）

    return {
        "states":  states,   # (T+1, 2)
        "actions": actions,  # (T, 1)
        "refs":    refs,     # (T, 1)
        "t_arr":   t_arr,    # (T,)
    }


# -----------------------------------------------------------------------
# 切分为 chunk
# -----------------------------------------------------------------------

def slice_into_chunks(
    traj: dict,
    H: int = 16,
) -> dict:
    """
    将一条轨迹切分为不重叠的 chunk，每个 chunk 长度为 H。

    Args:
        traj: collect_one_trajectory 的返回值
        H:    chunk 长度（prediction horizon）

    Returns:
        dict with:
            states:  (n_chunks, H+1, 2)   — 每个 chunk 的状态序列
            actions: (n_chunks, H, 1)     — 每个 chunk 的动作序列
            refs:    (n_chunks, H, 1)     — 每个 chunk 的参考序列
    """
    T = traj["actions"].shape[0]
    n_chunks = T // H

    chunk_states  = np.zeros((n_chunks, H + 1, 2), dtype=np.float32)
    chunk_actions = np.zeros((n_chunks, H,     1), dtype=np.float32)
    chunk_refs    = np.zeros((n_chunks, H,     1), dtype=np.float32)

    for i in range(n_chunks):
        start = i * H
        end   = start + H
        chunk_states[i]  = traj["states"][start : end + 1]   # (H+1, 2)
        chunk_actions[i] = traj["actions"][start : end]       # (H, 1)
        chunk_refs[i]    = traj["refs"][start : end]          # (H, 1)

    return {
        "states":  chunk_states,   # (n_chunks, H+1, 2)
        "actions": chunk_actions,  # (n_chunks, H, 1)
        "refs":    chunk_refs,     # (n_chunks, H, 1)
    }


# -----------------------------------------------------------------------
# 主采集函数
# -----------------------------------------------------------------------

def collect_dataset(
    n_trajs: int = 500,
    T_per_traj: int = 200,
    H: int = 16,
    dt: float = 0.05,
    seed: int = 0,
    save_path: str = "data/expert_dataset.npz",
) -> dict:
    """
    采集完整专家数据集。

    设计决策（见 CLAUDE.md §八）：
        - 500 条轨迹（250 阶跃 + 250 正弦），每条 200 步 → 12 chunks/traj → N=6000
        - 初始状态随机化：x_init ~ U(-0.5, 0.5)，x_dot_init ~ U(-0.5, 0.5)
          覆盖偏离状态，对 Exp-B 突发力实验的泛化性有帮助
        - 正弦频率范围 [0.05, 0.25] Hz（不超过系统固有频率 0.225 Hz）
        - obs = chunk 首帧 [x, x_dot]（不含参考位置）

    Args:
        n_trajs:     轨迹条数（一半阶跃，一半正弦）
        T_per_traj:  每条轨迹的步数
        H:           chunk 长度（prediction horizon）
        dt:          仿真步长
        seed:        随机种子
        save_path:   保存路径

    Returns:
        dataset dict with:
            states:  (N, H+1, 2)   N = n_trajs * (T_per_traj // H)
            actions: (N, H, 1)
            refs:    (N, H, 1)
    """
    rng = np.random.default_rng(seed)

    env = MSDEnv(dt=dt, seed=seed)
    pid = PIDController(dt=dt)

    all_states  = []
    all_actions = []
    all_refs    = []

    t_arr = np.arange(T_per_traj) * dt

    for i in range(n_trajs):
        # 随机初始状态（覆盖偏离状态，提升泛化性）
        x_init     = rng.uniform(-0.5, 0.5)
        x_dot_init = rng.uniform(-0.5, 0.5)
        env.reset(x_init=x_init, x_dot_init=x_dot_init)
        pid.reset()

        # 交替生成阶跃和正弦参考轨迹
        if i % 2 == 0:
            amp = rng.uniform(0.3, 1.5)
            ref_seq = make_step_ref(t_arr, amp=amp)
        else:
            amp   = rng.uniform(0.3, 1.2)
            # Frequency range capped at 0.25 Hz — MSD natural freq is 0.225 Hz,
            # beyond that the system cannot track and data quality degrades sharply.
            freq  = rng.uniform(0.05, 0.25)
            phase = rng.uniform(0, 2 * np.pi)
            ref_seq = make_sin_ref(t_arr, amp=amp, freq=freq, phase=phase)

        # 采集轨迹（env 已在上面 reset，这里不再重复 reset）
        traj = collect_one_trajectory_no_reset(env, pid, ref_seq, add_obs_noise=True)

        # 切分 chunk
        chunks = slice_into_chunks(traj, H=H)

        all_states.append(chunks["states"])    # (n_chunks, H+1, 2)
        all_actions.append(chunks["actions"])  # (n_chunks, H, 1)
        all_refs.append(chunks["refs"])        # (n_chunks, H, 1)

    # 拼接所有轨迹的 chunk
    # states:  (N, H+1, 2)
    # actions: (N, H, 1)
    # refs:    (N, H, 1)
    dataset = {
        "states":  np.concatenate(all_states,  axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "refs":    np.concatenate(all_refs,    axis=0),
    }

    N = dataset["states"].shape[0]
    print(f"Dataset collected: {N} chunks, H={H}")
    print(f"  states:  {dataset['states'].shape}")
    print(f"  actions: {dataset['actions'].shape}")
    print(f"  refs:    {dataset['refs'].shape}")

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **dataset)
    print(f"Saved to {save_path}")

    return dataset


# -----------------------------------------------------------------------
# 快速验证：直接运行此文件
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # --- Typography: serif font, no CJK issues ---
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
    })

    print("=== PID Expert Dataset Verification ===")

    # Small-scale collection for verification
    dataset = collect_dataset(
        n_trajs=10,
        T_per_traj=160,
        H=16,
        save_path="data/expert_dataset_debug.npz",
    )

    # Visualise one trajectory
    env = MSDEnv(seed=99)
    pid = PIDController()
    dt = 0.05
    T = 160
    t_arr = np.arange(T) * dt
    ref_seq = make_sin_ref(t_arr, amp=1.0, freq=0.5)

    env.reset()
    pid.reset()
    traj = collect_one_trajectory(env, pid, ref_seq, add_obs_noise=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(traj["t_arr"], ref_seq, "k--",
                 label=r"$x_{\mathrm{ref}}$", linewidth=1.5)
    axes[0].plot(traj["t_arr"], traj["states"][:-1, 0],
                 label=r"$x_{\mathrm{true}}$", linewidth=1.8)
    axes[0].set_ylabel(r"Position $x$ (m)")
    axes[0].legend(framealpha=0.9)
    axes[0].set_title("PID Expert Trajectory (sinusoidal reference, $f=0.5$ Hz)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(traj["t_arr"], traj["actions"][:, 0],
                 label=r"$u$ (N)", color="#E07B39", linewidth=1.5)
    axes[1].set_ylabel(r"Control force $u$ (N)")
    axes[1].set_xlabel(r"Time (s)")
    axes[1].legend(framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/pid_verify.png", dpi=150)
    print("Verification plot saved to data/pid_verify.png")
