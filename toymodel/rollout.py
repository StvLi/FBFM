"""
Closed-loop rollout functions for the 1D mass-spring-damper environment.

Provides two entry points:
  - run_rollout()      – flow-matching model with optional FBFM/RTC guidance
  - run_pid_rollout()  – deterministic PID expert baseline
"""

import torch

from pre_test_1.config import (
    STATE_DIM, ACTION_DIM, TOTAL_STEPS, REPLAN_INTERVAL, INFERENCE_DELAY,
)
from pre_test_1.physics_env import MassSpringDamperEnv
from pre_test_1.fbfm_processor import GuidanceMode, PrevChunkInfo, fbfm_sample
from pre_test_1.expert_data import PIDController, PIDConfig

# Max allowed action change per timestep.  PID expert's max slew ≈ 2.5;
# we use 3.0 to leave headroom while still killing ±10 spikes.
MAX_SLEW_RATE = 3.0


def run_rollout(model, norm_stats, cfg, env_cfg, target, init_state, device,
                disturbance=None):
    """Closed-loop rollout for 1D mass-spring-damper."""
    env = MassSpringDamperEnv(env_cfg) # create a new env instance for this rollout
    state = env.reset(init_state) # reset to the given initial state

    T = TOTAL_STEPS # total rollout length
    all_states = torch.zeros(T, STATE_DIM) # pre-allocate tensor to store states
    all_actions = torch.zeros(T, ACTION_DIM) # pre-allocate tensor to store actions
    pred_states_list = [] # list to store predicted states for debugging/analysis
    chunk_boundaries = [] # list to store indices where new chunks are sampled

    current_chunk = None # currently executing chunk of actions
    action_idx = 0 # index within the current chunk
    observed_states = [] # list to store states observed during the current chunk, for FBFM guidance

    t = 0
    while t < T:
        # Determine if we need to sample a new chunk (either at the start or after executing REPLAN_INTERVAL actions)
        need_new = (current_chunk is None or action_idx >= min(REPLAN_INTERVAL, current_chunk.shape[0]))
        if need_new:
            chunk_boundaries.append(t)
            prev_chunk = None
            if current_chunk is not None and cfg.mode != GuidanceMode.VANILLA:
                # leftover = current_chunk[action_idx:]
                # Leftover = chunk[4:11] (indices 4-10, 7 actions from the
                # actual execution segment).  These anchor the next chunk's
                # guided inference.  NOT chunk[9:] which includes empty tail.
                leftover = current_chunk[INFERENCE_DELAY:INFERENCE_DELAY + (cfg.horizon - INFERENCE_DELAY - cfg.empty_horizon)]
                prev_chunk = PrevChunkInfo(
                    action_leftover=leftover if leftover.shape[0] > 0 else None,
                    inference_delay=INFERENCE_DELAY,
                )
                if cfg.mode in (GuidanceMode.FBFM, GuidanceMode.FBFM_IDENTITY) and len(observed_states) > 0:
                    prev_chunk.observed_states = torch.stack(observed_states, dim=0)

            target_t = target[t].clone().detach()
            obs = state.clone().detach()
            pred_s, pred_a, _ = fbfm_sample(
                model=model, observation=obs, norm_stats=norm_stats,
                cfg=cfg, prev_chunk=prev_chunk, device=device,
                target=target_t,
            )
            pred_states_list.append(pred_s.detach())
            current_chunk = pred_a.detach().clamp(-env_cfg.max_force, env_cfg.max_force)
            action_idx = 0
            observed_states = []

        action = current_chunk[action_idx] if action_idx < current_chunk.shape[0] else torch.zeros(ACTION_DIM)

        # Slew-rate limiting: clamp change from previous executed action
        if t > 0:
            prev_action = all_actions[t - 1]
            delta = action - prev_action
            action = prev_action + delta.clamp(-MAX_SLEW_RATE * env_cfg.dt / env_cfg.dt,
                                                MAX_SLEW_RATE)
            # Also write back so the chunk stays consistent for leftover
            if action_idx < current_chunk.shape[0]:
                current_chunk[action_idx] = action

        all_states[t] = state.clone()
        all_actions[t] = action.clone()
        state = env.step(action.detach(), disturbance=disturbance)
        observed_states.append(state.clone().detach())
        action_idx += 1
        t += 1

    return {
        "states": all_states,
        "actions": all_actions,
        "targets": target[:T],
        "pred_states": pred_states_list,
        "chunk_boundaries": chunk_boundaries,
    }


def run_pid_rollout(env_cfg, target, init_state, disturbance=None):
    """PID expert rollout."""
    env = MassSpringDamperEnv(env_cfg)
    pid = PIDController(PIDConfig())
    state = env.reset(init_state)

    T = TOTAL_STEPS
    all_states = torch.zeros(T, STATE_DIM)
    all_actions = torch.zeros(T, ACTION_DIM)

    for t in range(T):
        u = pid.compute(state[0].item(), target[t, 0].item(), env_cfg.dt)
        action = torch.tensor([u])
        all_states[t] = state.clone()
        all_actions[t] = action.clone()
        state = env.step(action, disturbance=disturbance)

    return {"states": all_states, "actions": all_actions, "targets": target[:T]}
