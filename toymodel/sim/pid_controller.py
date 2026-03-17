"""
sim/pid_controller.py — PID 控制器

用于生成专家数据集。控制目标：使 MSD 系统的位置 x 跟踪参考轨迹 x_ref。

PID 公式：
    e(t)   = x_ref(t) - x(t)
    u(t)   = Kp * e(t) + Ki * integral(e) + Kd * de/dt

输入：
    state:  (2,)  — [x, x_dot]
    x_ref:  float — 当前时刻参考位置

输出：
    u:      float — 控制力 (N)，已 clip 到 [-u_max, u_max]
"""


class PIDController:
    """
    离散 PID 控制器（带积分饱和保护）。

    默认参数针对 MSD(m=1, k=2, c=0.5, dt=0.05) 调参，
    可跟踪阶跃和正弦参考轨迹。
    """

    def __init__(
        self,
        Kp: float = 30.0,
        Ki: float = 15.0,
        Kd: float = 5.0,
        dt: float = 0.05,
        u_max: float = 20.0,
        integral_max: float = 10.0,
    ):
        """
        Args:
            Kp:           比例增益
            Ki:           积分增益
            Kd:           微分增益
            dt:           控制步长 (s)，需与 MSDEnv.dt 一致
            u_max:        控制力饱和上限 (N)
            integral_max: 积分项饱和上限（防积分饱和）
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_max = u_max
        self.integral_max = integral_max

        # 内部状态
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def reset(self):
        """重置积分项和上一步误差。"""
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, state, x_ref: float) -> float:
        """
        计算控制力。

        Args:
            state:  (2,) — [x, x_dot]，当前观测状态
            x_ref:  float — 当前参考位置

        Returns:
            u: float — 控制力 (N)
        """
        x = float(state[0])

        # 误差
        error = x_ref - x

        # 积分项（带饱和保护）
        self._integral += error * self.dt
        self._integral = float(
            max(-self.integral_max, min(self.integral_max, self._integral))
        )

        # 微分项（后向差分）
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        # PID 输出
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # 饱和限幅
        u = float(max(-self.u_max, min(self.u_max, u)))
        return u
