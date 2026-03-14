"""
1D Mass-Spring-Damper Physics Environment for FBFM Pre-experiment.

System: m*x'' + c*x' + k*x = u + Δ
State:  s = [x, x']  (position, velocity)
Action: a = [u]       (applied force)
Δ:      external disturbance (step / impulse)

The environment simulates a second-order oscillator controlled by an external
force u. This serves as a minimal test-bed for validating the FBFM state
feedback algorithm before deploying on real robotic systems.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor


@dataclass
class EnvConfig:
    """Physics parameters for the 1D oscillator."""

    mass: float = 1.0
    damping: float = 0.5
    stiffness: float = 0.1
    dt: float = 0.02          # 50 Hz control frequency
    max_force: float = 10.0   # clamp applied force
    process_noise_std: float = 0.0  # Continuous process noise (random force disturbance)


@dataclass
class Disturbance:
    """Specification for an external disturbance injected during rollout.

    Attributes:
        start_step: physical step at which disturbance begins.
        duration: number of steps it lasts (0 = impulse at start_step only).
        magnitude: force magnitude of the disturbance.
        type: 'step' (持续力偏移) or 'impulse' (单步冲击) or 'position_offset'
              (直接位移偏移, 模拟被推了一下).
    """

    start_step: int = 50
    duration: int = 0        # 0 → single-step impulse
    magnitude: float = 2.0
    type: str = "position_offset"  # step | impulse | position_offset


class MassSpringDamperEnv:
    """1D mass-spring-damper simulation.

    The state is [x, x_dot]. Semi-implicit Euler integration is used for
    numerical stability at the chosen dt.
    """

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = cfg or EnvConfig()
        self.state: Tensor = torch.zeros(2)  # [x, x_dot]
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self, init_state: Tensor | None = None) -> Tensor:
        """Reset the environment.

        Args:
            init_state: optional (2,) tensor [x0, x_dot_0]. Defaults to zeros.

        Returns:
            Current state (2,).
        """
        if init_state is not None:
            self.state = init_state.clone().float()
        else:
            self.state = torch.zeros(2)
        self.step_count = 0
        return self.state.clone()

    def step(
        self,
        action: Tensor,
        disturbance: Disturbance | None = None,
    ) -> Tensor:
        """Advance the simulation by one timestep.

        Args:
            action: (1,) or scalar – applied force u.
            disturbance: optional external disturbance specification.

        Returns:
            New state (2,).
        """
        m, c, k, dt = self.cfg.mass, self.cfg.damping, self.cfg.stiffness, self.cfg.dt
        u = torch.clamp(action.float().squeeze(), -self.cfg.max_force, self.cfg.max_force)

        x, x_dot = self.state[0], self.state[1]

        # External disturbance force (additive)
        delta_force = 0.0
        apply_pos_offset = False
        pos_offset_val = 0.0

        if disturbance is not None:
            in_window = (
                self.step_count >= disturbance.start_step
                and (
                    disturbance.duration == 0
                    and self.step_count == disturbance.start_step
                )
                or (
                    disturbance.duration > 0
                    and disturbance.start_step
                    <= self.step_count
                    < disturbance.start_step + disturbance.duration
                )
            )
            if in_window:
                if disturbance.type == "step":
                    delta_force = disturbance.magnitude
                elif disturbance.type == "impulse":
                    delta_force = disturbance.magnitude
                elif disturbance.type == "position_offset":
                    # Directly shift position (simulates being pushed)
                    if self.step_count == disturbance.start_step:
                        apply_pos_offset = True
                        pos_offset_val = disturbance.magnitude

        # Semi-implicit Euler
        x_ddot = (u + delta_force - c * x_dot - k * x) / m

        # Add continuous process noise (random force perturbation)
        if self.cfg.process_noise_std > 0:
            noise_force = torch.randn(1).item() * self.cfg.process_noise_std
            x_ddot = x_ddot + noise_force / m

        x_dot_new = x_dot + x_ddot * dt
        x_new = x + x_dot_new * dt

        if apply_pos_offset:
            x_new = x_new + pos_offset_val

        self.state = torch.tensor([x_new, x_dot_new], dtype=torch.float32)
        self.step_count += 1
        return self.state.clone()

    def get_state(self) -> Tensor:
        """Return current state (2,)."""
        return self.state.clone()


# ------------------------------------------------------------------
# Target trajectory generators
# ------------------------------------------------------------------

def generate_step_target(total_steps: int, target_pos: float = 1.0) -> Tensor:
    """Generate a step-response target: 0 → target_pos at step 0.

    Returns:
        Tensor of shape (total_steps, 2) – target [x, x_dot] per step.
        x_dot target is always 0 (we want zero velocity at steady state).
    """
    targets = torch.zeros(total_steps, 2)
    targets[:, 0] = target_pos
    return targets


def generate_sinusoidal_target(
    total_steps: int,
    amplitude: float = 1.0,
    period_steps: int = 200,
) -> Tensor:
    """Generate a sinusoidal tracking target.

    Returns:
        Tensor of shape (total_steps, 2) – target [x, x_dot].
    """
    t = torch.arange(total_steps, dtype=torch.float32)
    omega = 2.0 * torch.pi / period_steps
    x_target = amplitude * torch.sin(omega * t)
    xdot_target = amplitude * omega * torch.cos(omega * t)
    return torch.stack([x_target, xdot_target], dim=-1)


def generate_random_setpoint_target(
    total_steps: int,
    change_interval: int = 100,
    pos_range: tuple = (-2.0, 2.0),
) -> Tensor:
    """Generate a random piece-wise constant setpoint trajectory.

    Returns:
        Tensor of shape (total_steps, 2) – target [x, x_dot=0].
    """
    targets = torch.zeros(total_steps, 2)
    current_pos = 0.0
    for i in range(0, total_steps, change_interval):
        current_pos = torch.empty(1).uniform_(*pos_range).item()
        end = min(i + change_interval, total_steps)
        targets[i:end, 0] = current_pos
    return targets

def generate_ramp_target(
    total_steps: int,
    slope: float = 0.01,
    max_val: float = 2.0
) -> Tensor:
    """Generate a ramp target that increases linearly up to a max value.
    
    Returns:
        Tensor of shape (total_steps, 2) - target [x, x_dot].
    """
    t = torch.arange(total_steps, dtype=torch.float32)
    x_target = torch.clamp(slope * t, -max_val, max_val)
    xdot_target = torch.where(torch.abs(x_target) < max_val, torch.tensor(slope), torch.tensor(0.0))
    return torch.stack([x_target, xdot_target], dim=-1)

def generate_multi_sin_target(
    total_steps: int,
    amplitudes: tuple[float, ...] = (1.0, 0.5),
    period_steps: tuple[int, ...] = (200, 50),
) -> Tensor:
    """Generate a target built from multiple summed sinusoids."""
    t = torch.arange(total_steps, dtype=torch.float32)
    x_target = torch.zeros(total_steps, dtype=torch.float32)
    xdot_target = torch.zeros(total_steps, dtype=torch.float32)
    
    for amp, p in zip(amplitudes, period_steps):
        omega = 2.0 * torch.pi / p
        x_target += amp * torch.sin(omega * t)
        xdot_target += amp * omega * torch.cos(omega * t)
        
    return torch.stack([x_target, xdot_target], dim=-1)
