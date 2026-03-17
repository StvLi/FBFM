"""
sim/msd_env.py — 一维质量-弹簧-阻尼器 (Mass-Spring-Damper, MSD) 仿真环境

物理模型：
    m * x_ddot = u - k*x - c*x_dot + w_process

状态量 s = [x, x_dot]  shape: (2,)
控制量 a = [u]          shape: (1,) 或 scalar

扰动噪声添加位置（见 CLAUDE.md §2.3）：
    1. 过程噪声 w_process：在动力学积分内部，模拟建模误差
    2. 观测噪声 w_obs：    在 step() 返回 state 时叠加，模拟传感器噪声
    3. 外部扰动力：        通过 step(u, disturbance=F_ext) 显式传入
"""

import numpy as np


class MSDEnv:
    """
    一维质量-弹簧-阻尼器环境。

    默认参数（见 CLAUDE.md §六）：
        m=1.0 kg, k=2.0 N/m, c=0.5 N·s/m, dt=0.05 s

    状态空间：
        state: (2,)  — [x (m), x_dot (m/s)]

    动作空间：
        action: scalar or (1,)  — u (N), 施加在质量块上的力

    噪声参数：
        obs_noise_std:     观测噪声标准差（默认 0.01）
        process_noise_std: 过程噪声标准差（默认 0.005）
    """

    def __init__(
        self,
        m: float = 1.0,
        k: float = 2.0,
        c: float = 0.5,
        dt: float = 0.05,
        obs_noise_std: float = 0.01,
        process_noise_std: float = 0.005,
        x_init: float = 0.0,
        x_dot_init: float = 0.0,
        seed: int = 42,
    ):
        """
        Args:
            m:                 质量 (kg)
            k:                 弹簧刚度 (N/m)
            c:                 阻尼系数 (N·s/m)
            dt:                仿真步长 (s)
            obs_noise_std:     观测噪声标准差
            process_noise_std: 过程噪声标准差
            x_init:            初始位置 (m)
            x_dot_init:        初始速度 (m/s)
            seed:              随机种子，保证可复现
        """
        # 物理参数
        self.m = m
        self.k = k
        self.c = c
        self.dt = dt

        # 噪声参数
        self.obs_noise_std = obs_noise_std
        self.process_noise_std = process_noise_std

        # 初始条件
        self.x_init = x_init
        self.x_dot_init = x_dot_init

        # 随机数生成器（固定种子保证可复现）
        self.rng = np.random.default_rng(seed)

        # 内部状态
        # state: (2,) — [x, x_dot]
        self.state: np.ndarray = np.array([x_init, x_dot_init], dtype=np.float64)
        self.t: float = 0.0          # 当前仿真时间 (s)
        self.step_count: int = 0     # 已执行步数

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def reset(self, x_init: float = None, x_dot_init: float = None) -> np.ndarray:
        """
        重置环境到初始状态。

        Args:
            x_init:     覆盖初始位置（None 则使用构造时的值）
            x_dot_init: 覆盖初始速度（None 则使用构造时的值）

        Returns:
            state: (2,) — [x, x_dot]，不含观测噪声（干净初始状态）
        """
        x0 = x_init if x_init is not None else self.x_init
        v0 = x_dot_init if x_dot_init is not None else self.x_dot_init

        # state: (2,)
        self.state = np.array([x0, v0], dtype=np.float64)
        self.t = 0.0
        self.step_count = 0
        return self.state.copy()

    def step(
        self,
        u: float,
        disturbance: float = 0.0,
        add_obs_noise: bool = True,
    ):
        """
        执行一步仿真（Euler 积分）。

        物理方程：
            x_ddot = (u + disturbance - k*x - c*x_dot + w_process) / m

        噪声注入：
            - 过程噪声 w_process ~ N(0, process_noise_std²)：在积分内部
            - 观测噪声 w_obs     ~ N(0, obs_noise_std²)：在返回 state 时

        Args:
            u:             控制力 (N)，scalar
            disturbance:   外部扰动力 (N)，scalar，默认 0
                           用于 Exp-B 突发力实验（见 CLAUDE.md §5.1）
            add_obs_noise: 是否在返回的 state 上叠加观测噪声

        Returns:
            obs:   (2,) — 含观测噪声的状态 [x_obs, x_dot_obs]
            reward: float — 负的位置误差平方（占位，后续可扩展）
            done:   bool  — 本环境无终止条件，始终 False
            info:   dict  — 包含真实状态 true_state 和当前时间 t
        """
        x, x_dot = self.state  # 当前真实状态

        # ---- 过程噪声（模拟建模误差）----
        # w_process: scalar ~ N(0, process_noise_std²)
        w_process = self.rng.normal(0.0, self.process_noise_std)

        # ---- 动力学方程（Euler 积分）----
        # x_ddot: scalar (m/s²)
        x_ddot = (u + disturbance - self.k * x - self.c * x_dot + w_process) / self.m

        # 更新状态
        x_new     = x     + self.dt * x_dot
        x_dot_new = x_dot + self.dt * x_ddot

        # state: (2,)
        self.state = np.array([x_new, x_dot_new], dtype=np.float64)
        self.t += self.dt
        self.step_count += 1

        # ---- 观测噪声（模拟传感器噪声）----
        # w_obs: (2,) ~ N(0, obs_noise_std²)
        if add_obs_noise:
            w_obs = self.rng.normal(0.0, self.obs_noise_std, size=2)
        else:
            w_obs = np.zeros(2)

        # obs: (2,) — 控制器实际能读到的带噪声状态
        obs = self.state + w_obs

        # 简单奖励：负位置误差平方（占位）
        reward = -float(x_new ** 2)

        done = False

        info = {
            "true_state": self.state.copy(),  # (2,) 真实状态（无噪声）
            "t": self.t,
            "step": self.step_count,
            "w_process": w_process,
            "disturbance": disturbance,
        }

        return obs, reward, done, info

    def get_params(self) -> dict:
        """返回当前物理参数，便于实验记录和参数失配测试。"""
        return {"m": self.m, "k": self.k, "c": self.c, "dt": self.dt}

    def set_params(self, m: float = None, k: float = None, c: float = None):
        """
        在线修改物理参数，用于 Exp-A 参数失配实验（见 CLAUDE.md §5.1）。

        Args:
            m: 新质量 (kg)，None 表示不修改
            k: 新弹簧刚度 (N/m)，None 表示不修改
            c: 新阻尼系数 (N·s/m)，None 表示不修改
        """
        if m is not None:
            self.m = m
        if k is not None:
            self.k = k
        if c is not None:
            self.c = c

    def __repr__(self) -> str:
        return (
            f"MSDEnv(m={self.m}, k={self.k}, c={self.c}, dt={self.dt}) "
            f"| t={self.t:.3f}s, state=[x={self.state[0]:.4f}, x_dot={self.state[1]:.4f}]"
        )


# ------------------------------------------------------------------
# 快速验证：直接运行此文件即可检查环境是否正常工作
# ------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # --- Typography: serif font throughout, no CJK fallback issues ---
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":  "stix",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })

    print("=== MSD Environment Verification ===")

    env = MSDEnv(seed=0)
    print(f"Init: {env}")
    print(f"Params: {env.get_params()}")

    # Apply constant force u=1 N and observe system response
    T = 5.0          # simulation duration (s)
    n_steps = int(T / env.dt)
    u_const = 1.0    # constant control force (N)

    times       = []   # time series
    xs_true     = []   # true position
    xs_obs      = []   # observed position (with noise)
    x_dots_true = []   # true velocity

    obs = env.reset()
    for i in range(n_steps):
        # Apply impulse disturbance at step 50 to verify disturbance interface
        dist = 5.0 if i == 50 else 0.0
        obs, reward, done, info = env.step(u_const, disturbance=dist)

        times.append(info["t"])
        xs_true.append(info["true_state"][0])
        xs_obs.append(obs[0])
        x_dots_true.append(info["true_state"][1])

    print(f"Simulation done: {n_steps} steps, final state: {env}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(times, xs_true, label=r"$x_{\mathrm{true}}$", linewidth=1.8)
    axes[0].plot(times, xs_obs,  label=r"$x_{\mathrm{obs}}$ (noisy)",
                 linewidth=1.0, alpha=0.7, linestyle="--")
    axes[0].axvline(x=50 * env.dt, color="crimson", linestyle=":",
                    linewidth=1.5, label="Impulse disturbance $t=2.5$ s")
    axes[0].set_ylabel(r"Position $x$ (m)")
    axes[0].legend(framealpha=0.9)
    axes[0].set_title(r"MSD Environment Verification: constant force $u=1\,\mathrm{N}$ + impulse disturbance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, x_dots_true, label=r"$\dot{x}_{\mathrm{true}}$",
                 linewidth=1.8, color="#E07B39")
    axes[1].set_ylabel(r"Velocity $\dot{x}$ (m/s)")
    axes[1].set_xlabel(r"Time (s)")
    axes[1].legend(framealpha=0.9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sim/msd_env_verify.png", dpi=150)
    print("Verification plot saved to sim/msd_env_verify.png")
    plt.show()
