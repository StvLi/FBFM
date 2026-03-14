"""
Inference pipeline for 1D FBFM pre-experiment.

Implements a chunk-based closed-loop rollout that:
  1. Generates an action chunk via flow-matching (vanilla / RTC / FBFM).
  2. Executes chunk_execution_horizon steps in the physics environment.
  3. Collects real state observations during execution.
  4. Feeds back leftover actions + observed states into the next chunk generation.

This simulates the real-time execution ↔ inference interleaving described in
guidance.md §3 and Algorithm 1 of core_thoughts.md, but uses a synchronous
(single-thread) implementation for clarity and reproducibility.

For the delay compensation test, we simulate inference delay by skipping
the first `inference_delay` actions of each new chunk.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from pre_test_2.physics_env import MassSpringDamperEnv, EnvConfig, Disturbance
from pre_test_2.fbfm_processor import (
    FBFMConfig,
    FBFMProcessor,
    PrevChunkInfo,
    GuidanceMode,
    fbfm_sample,
)
from pre_test_2.expert_data import PIDController, PIDConfig


@dataclass
class RolloutConfig:
    """Configuration for a closed-loop rollout experiment."""

    total_steps: int = 300
    chunk_execution_horizon: int = 8   # s_chunk: how many steps to execute per chunk
    inference_delay: int = 0           # simulated inference delay (steps)
    horizon: int = 16                  # prediction horizon H
    disturbance: Disturbance | None = None  # optional external disturbance


@dataclass
class RolloutResult:
    """Container for rollout results."""

    states: Tensor           # (T, state_dim) – actual trajectory
    actions: Tensor          # (T, action_dim) – applied actions
    targets: Tensor          # (T, state_dim) – target trajectory
    predicted_states: list   # list of (H, state_dim) – predicted state chunks
    predicted_actions: list  # list of (H, action_dim) – predicted action chunks
    chunk_boundaries: list   # list of int – step indices where new chunks started
    debug_all: list          # all debug info from FBFM processor


def run_rollout(
    model,
    norm_stats: dict,
    fbfm_cfg: FBFMConfig,
    rollout_cfg: RolloutConfig,
    env_cfg: EnvConfig,
    target_trajectory: Tensor,
    init_state: Tensor | None = None,
    device: str = "cpu",
) -> RolloutResult:
    """Execute a closed-loop rollout with chunk-based inference.

    This is the main experiment runner. It simulates the execution-inference
    interleaving described in guidance.md §3.1 and Algorithm 1 of core_thoughts.md.

    The flow is:
      1. Generate a new action chunk via flow-matching.
      2. Execute ``chunk_execution_horizon`` steps from that chunk in the env.
      3. During execution, collect the *real physical states* observed at each step.
      4. When requesting the next chunk, package:
           - action_leftover: unexecuted tail of the current chunk
           - observed_states: real states collected so far during this chunk
         into PrevChunkInfo for guided inference.

    Args:
        model: trained FlowMatchingDiT.
        norm_stats: normalization statistics from training.
        fbfm_cfg: FBFM configuration (determines vanilla/RTC/FBFM mode).
        rollout_cfg: rollout parameters.
        env_cfg: physics environment parameters.
        target_trajectory: (total_steps, state_dim) – desired trajectory.
        init_state: optional initial state.
        device: torch device.

    Returns:
        RolloutResult with all trajectory data.
    """
    env = MassSpringDamperEnv(env_cfg)
    state = env.reset(init_state)

    T = rollout_cfg.total_steps
    H = rollout_cfg.horizon
    exec_h = rollout_cfg.chunk_execution_horizon
    delay = rollout_cfg.inference_delay

    state_dim = fbfm_cfg.state_dim
    action_dim = fbfm_cfg.action_dim
    max_force = env_cfg.max_force

    # Storage
    all_states = torch.zeros(T, state_dim)
    all_actions = torch.zeros(T, action_dim)
    pred_states_list = []
    pred_actions_list = []
    chunk_boundaries = []
    all_debug = []

    # Chunk execution state (local variables, no function attributes)
    current_action_chunk = None   # (H, action_dim) – full predicted actions for current chunk
    action_idx = 0                # how many steps we have executed from current_action_chunk
    observed_states_this_chunk: list[Tensor] = []  # states observed while executing current chunk

    t = 0
    while t < T:
        # Check if we need a new chunk
        need_new_chunk = (
            current_action_chunk is None
            or action_idx >= min(exec_h, current_action_chunk.shape[0])
        )

        if need_new_chunk:
            chunk_boundaries.append(t)

            # ---- Build PrevChunkInfo from the CURRENT (about-to-be-replaced) chunk ----
            prev_chunk_info = None
            if current_action_chunk is not None and fbfm_cfg.mode != GuidanceMode.VANILLA:
                # Leftover = unexecuted actions from current chunk
                leftover_actions = current_action_chunk[action_idx:]  # (remaining, action_dim)

                prev_chunk_info = PrevChunkInfo(
                    action_leftover=leftover_actions if leftover_actions.shape[0] > 0 else None,
                    inference_delay=delay,
                )

                # For FBFM / FBFM_IDENTITY: attach observed states collected during this chunk's execution
                if fbfm_cfg.mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY) and len(observed_states_this_chunk) > 0:
                    prev_chunk_info.observed_states = torch.stack(
                        observed_states_this_chunk, dim=0
                    )  # (num_executed, state_dim)

            # ---- Build target for model conditioning ----
            # Use the target at the current timestep
            target_t = target_trajectory[t].clone().detach()  # (state_dim,)

            # ---- Generate new chunk ----
            observation = state.clone().detach()
            pred_s, pred_a, debug = fbfm_sample(
                model=model,
                observation=observation,
                norm_stats=norm_stats,
                cfg=fbfm_cfg,
                prev_chunk=prev_chunk_info,
                device=device,
                target=target_t,
            )

            pred_states_list.append(pred_s.detach())
            pred_actions_list.append(pred_a.detach())
            all_debug.extend(debug)

            # Apply inference delay: skip first `delay` actions
            raw_actions = pred_a[delay:].detach()  # (H - delay, action_dim)
            # Clamp actions to physical limits to prevent saturation cascades
            current_action_chunk = raw_actions.clamp(-max_force, max_force)
            action_idx = 0
            observed_states_this_chunk = []  # reset for new chunk

        # ---- Execute one step ----
        if current_action_chunk is not None and action_idx < current_action_chunk.shape[0]:
            action = current_action_chunk[action_idx]
        else:
            action = torch.zeros(action_dim)

        all_states[t] = state.clone().detach()
        all_actions[t] = action.clone().detach()

        # Step environment
        state = env.step(action.detach(), disturbance=rollout_cfg.disturbance)

        # Collect the NEW observed state for potential FBFM feedback in next chunk
        observed_states_this_chunk.append(state.clone().detach())

        action_idx += 1
        t += 1

    return RolloutResult(
        states=all_states,
        actions=all_actions,
        targets=target_trajectory[:T],
        predicted_states=pred_states_list,
        predicted_actions=pred_actions_list,
        chunk_boundaries=chunk_boundaries,
        debug_all=all_debug,
    )


def run_pid_rollout(
    rollout_cfg: RolloutConfig,
    env_cfg: EnvConfig,
    target_trajectory: Tensor,
    pid_cfg: PIDConfig | None = None,
    init_state: Tensor | None = None,
) -> RolloutResult:
    """Run a PID expert rollout for reference comparison.

    Args:
        rollout_cfg: rollout parameters.
        env_cfg: physics environment parameters.
        target_trajectory: (total_steps, state_dim) – desired trajectory.
        pid_cfg: PID gains. Defaults to tuned gains.
        init_state: optional initial state.

    Returns:
        RolloutResult with PID trajectory.
    """
    env = MassSpringDamperEnv(env_cfg)
    pid = PIDController(pid_cfg or PIDConfig())
    state = env.reset(init_state)

    T = rollout_cfg.total_steps
    state_dim = 2
    action_dim = 1

    all_states = torch.zeros(T, state_dim)
    all_actions = torch.zeros(T, action_dim)

    for t in range(T):
        target_pos = target_trajectory[t, 0].item()
        u = pid.compute(state[0].item(), target_pos, env_cfg.dt)
        action = torch.tensor([u])

        all_states[t] = state.clone()
        all_actions[t] = action.clone()

        state = env.step(action, disturbance=rollout_cfg.disturbance)

    return RolloutResult(
        states=all_states,
        actions=all_actions,
        targets=target_trajectory[:T],
        predicted_states=[],
        predicted_actions=[],
        chunk_boundaries=[],
        debug_all=[],
    )


# ======================================================================
# Evaluation metrics
# ======================================================================

def compute_metrics(result: RolloutResult, reference: RolloutResult | None = None) -> dict:
    """Compute evaluation metrics for a rollout.

    Args:
        result: rollout result to evaluate.
        reference: optional PID expert reference for comparison.

    Returns:
        Dictionary of metrics.
    """
    metrics = {}

    # State MSE vs target
    state_error = result.states - result.targets
    metrics["state_mse"] = (state_error ** 2).mean().item()
    metrics["position_mse"] = (state_error[:, 0] ** 2).mean().item()
    metrics["velocity_mse"] = (state_error[:, 1] ** 2).mean().item()

    # Action MSE vs PID reference (if available)
    if reference is not None:
        action_error = result.actions - reference.actions
        metrics["action_mse_vs_pid"] = (action_error ** 2).mean().item()
        state_error_vs_pid = result.states - reference.states
        metrics["state_mse_vs_pid"] = (state_error_vs_pid ** 2).mean().item()

    # Action jitter (consecutive action difference variance)
    if result.actions.shape[0] > 1:
        action_diff = result.actions[1:] - result.actions[:-1]
        metrics["action_jitter"] = action_diff.var().item()

    # Recovery steps (if there's a disturbance)
    # Find peak error and measure steps to return below threshold
    pos_error = (result.states[:, 0] - result.targets[:, 0]).abs()
    if pos_error.max() > 0.5:  # significant disturbance occurred
        peak_idx = pos_error.argmax().item()
        threshold = 0.1  # recovery threshold
        recovery_steps = None
        for t in range(peak_idx, len(pos_error)):
            if pos_error[t] < threshold:
                recovery_steps = t - peak_idx
                break
        metrics["recovery_steps"] = recovery_steps
        metrics["peak_error"] = pos_error.max().item()
        metrics["peak_error_step"] = peak_idx

    return metrics
