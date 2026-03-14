"""
Expert data generation for the 1D FBFM pre-experiment.

Uses a PID controller as the expert policy to generate demonstration
trajectories. The data format matches the convention in configuration_rtc.py:

  chunk X = [s_0, s_1, ..., s_{H-1}, a_0, a_1, ..., a_{H-1}]

where s_t is the 2D state [x, x_dot] and a_t is the 1D action [u].
So each chunk is (H, state_dim + action_dim) = (H, 3).

The dataset stores:
  - observations: (N, state_dim)        current observation o_t
  - chunks:       (N, H, state_dim + action_dim)  future chunk X
  - targets:      (N, H, state_dim)      target states for reference
"""

import os
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from pre_test_2.physics_env import (
    MassSpringDamperEnv,
    EnvConfig,
    generate_step_target,
    generate_sinusoidal_target,
    generate_random_setpoint_target,
)


# ======================================================================
# PID Expert Controller
# ======================================================================

@dataclass
class PIDConfig:
    """PID gains for the expert controller."""

    kp: float = 8.0
    ki: float = 0.5
    kd: float = 4.0
    max_output: float = 10.0


class PIDController:
    """Simple PID controller for 1D position tracking."""

    def __init__(self, cfg: PIDConfig | None = None):
        self.cfg = cfg or PIDConfig()
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_pos: float, target_pos: float, dt: float) -> float:
        """Compute PID control output.

        Args:
            current_pos: current position x.
            target_pos: desired position.
            dt: timestep.

        Returns:
            Control force u.
        """
        error = target_pos - current_pos
        self.integral += error * dt
        # Anti-windup
        self.integral = max(-self.cfg.max_output, min(self.cfg.max_output, self.integral))
        derivative = (error - self.prev_error) / (dt + 1e-8)
        self.prev_error = error

        output = self.cfg.kp * error + self.cfg.ki * self.integral + self.cfg.kd * derivative
        return max(-self.cfg.max_output, min(self.cfg.max_output, output))


# ======================================================================
# Dataset generation
# ======================================================================

def collect_expert_trajectory(
    env_cfg: EnvConfig,
    pid_cfg: PIDConfig,
    target_fn,
    total_steps: int,
    init_state: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rollout PID expert in the 1D environment.

    Returns:
        states:  (total_steps, 2)
        actions: (total_steps, 1)
        targets: (total_steps, 2)
    """
    env = MassSpringDamperEnv(env_cfg)
    pid = PIDController(pid_cfg)
    state = env.reset(init_state)

    targets = target_fn(total_steps)

    states_list = []
    actions_list = []

    for t in range(total_steps):
        target_pos = targets[t, 0].item()
        u = pid.compute(state[0].item(), target_pos, env_cfg.dt)
        action = torch.tensor([u])
        states_list.append(state.clone())
        actions_list.append(action.clone())
        state = env.step(action)

    states = torch.stack(states_list, dim=0)   # (T, 2)
    actions = torch.stack(actions_list, dim=0)  # (T, 1)
    return states, actions, targets


def build_chunks_from_trajectory(
    states: Tensor,
    actions: Tensor,
    targets: Tensor,
    horizon: int = 16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Slice a trajectory into overlapping chunks.

    For each valid starting index t, extract:
      observation: states[t]                    → (state_dim,)
      chunk X:     [states[t:t+H], actions[t:t+H]] → (H, state_dim + action_dim)
      target:      targets[t:t+H]              → (H, state_dim)

    Returns:
        observations: (N, state_dim)
        chunks:       (N, H, state_dim + action_dim)
        chunk_targets: (N, H, state_dim)
    """
    T = states.shape[0]
    obs_list, chunk_list, tgt_list = [], [], []

    for t in range(T - horizon):
        obs_list.append(states[t])
        # Layout: [state; action] per timestep → (H, 3)
        chunk = torch.cat([states[t : t + horizon], actions[t : t + horizon]], dim=-1)
        chunk_list.append(chunk)
        tgt_list.append(targets[t : t + horizon])

    observations = torch.stack(obs_list, dim=0)
    chunks = torch.stack(chunk_list, dim=0)
    chunk_targets = torch.stack(tgt_list, dim=0)
    return observations, chunks, chunk_targets


class ExpertChunkDataset(Dataset):
    """PyTorch Dataset wrapping expert demonstration chunks."""

    def __init__(self, observations: Tensor, chunks: Tensor, targets: Tensor):
        assert observations.shape[0] == chunks.shape[0] == targets.shape[0]
        self.observations = observations
        self.chunks = chunks
        self.targets = targets

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        item = {
            "observation": self.observations[idx],
            "chunk": self.chunks[idx],
            "target": self.targets[idx],
        }
        # If normalized targets are stored, include them
        if hasattr(self, '_normalized_targets') and self._normalized_targets is not None:
            item["target_goal"] = self._normalized_targets[idx]
        else:
            # Fallback: use first timestep of raw target
            item["target_goal"] = self.targets[idx, 0, :]
        return item


def generate_dataset(
    num_trajectories: int = 500,
    steps_per_traj: int = 300,
    horizon: int = 16,
    env_cfg: EnvConfig | None = None,
    pid_cfg: PIDConfig | None = None,
    seed: int = 42,
) -> ExpertChunkDataset:
    """Generate a full expert dataset with diverse target trajectories.

    Args:
        num_trajectories: number of rollout episodes.
        steps_per_traj: physical steps per episode.
        horizon: chunk length H.
        env_cfg: physics parameters.
        pid_cfg: PID gains.
        seed: random seed.

    Returns:
        ExpertChunkDataset containing all chunks.
    """
    torch.manual_seed(seed)
    env_cfg = env_cfg or EnvConfig()
    pid_cfg = pid_cfg or PIDConfig()

    all_obs, all_chunks, all_targets = [], [], []

    target_generators = [
        # Step responses with random amplitudes
        lambda total: generate_step_target(total, target_pos=torch.empty(1).uniform_(0.3, 2.5).item()),
        # Sinusoidal tracking with random amplitude/period
        lambda total: generate_sinusoidal_target(
            total,
            amplitude=torch.empty(1).uniform_(0.5, 2.0).item(),
            period_steps=int(torch.empty(1).uniform_(100, 400).item()),
        ),
        # Random setpoint switching
        lambda total: generate_random_setpoint_target(
            total,
            change_interval=int(torch.empty(1).uniform_(50, 150).item()),
            pos_range=(-2.0, 2.0),
        ),
    ]

    for i in range(num_trajectories):
        # Randomly pick a target generator
        gen = target_generators[i % len(target_generators)]

        # Random initial state (small perturbation)
        init_state = torch.randn(2) * 0.3

        states, actions, targets = collect_expert_trajectory(
            env_cfg, pid_cfg, gen, steps_per_traj, init_state
        )
        obs, chunks, tgts = build_chunks_from_trajectory(states, actions, targets, horizon)
        all_obs.append(obs)
        all_chunks.append(chunks)
        all_targets.append(tgts)

    observations = torch.cat(all_obs, dim=0)
    chunks = torch.cat(all_chunks, dim=0)
    targets = torch.cat(all_targets, dim=0)

    print(f"[DataGen] Generated {observations.shape[0]} chunks from {num_trajectories} trajectories")
    print(f"  observation shape: {observations.shape}")
    print(f"  chunk shape:       {chunks.shape}")
    print(f"  target shape:      {targets.shape}")

    return ExpertChunkDataset(observations, chunks, targets)


def save_dataset(dataset: ExpertChunkDataset, path: str):
    """Save dataset to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "observations": dataset.observations,
            "chunks": dataset.chunks,
            "targets": dataset.targets,
        },
        path,
    )
    print(f"[DataGen] Saved dataset to {path}")


def load_dataset(path: str) -> ExpertChunkDataset:
    """Load dataset from disk."""
    data = torch.load(path, weights_only=True)
    return ExpertChunkDataset(data["observations"], data["chunks"], data["targets"])
