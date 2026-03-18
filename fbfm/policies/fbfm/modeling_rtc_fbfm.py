#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-Time Chunking (RTC) implementation for LeRobot.

Based on Physical Intelligence's Kinetix implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214

Based on LeRobot's RTC implementation:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py


┌─────────────────────────────────────────────────────────────────┐
│                    FBFM.PrevChunk 状态管理                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Chunk N-1                    Chunk N                           │
│  ┌──────────┐                ┌──────────┐                       │
│  │ 生成动作  │ ──────────────▶│ 接收反馈  │                      │
│  └──────────┘   (actions)    └──────────┘                       │
│       │                          │                              │
│       ▼                          ▼                              │
│  ┌──────────┐                ┌──────────┐                       │
│  │ 状态观测  │ ──────────────▶│ 约束注入  │                      │
│  └──────────┘   (obs/latent) └──────────┘                       │
│                                  │                              │
│                                  ▼                              │
│                          ┌──────────────┐                       │
│                          │ Scheduler.step│                      │
│                          │ constrained_y │                      │
│                          │ weights       │                      │
│                          └──────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from fbfm.configs.types import RTCAttentionSchedule
from fbfm.policies.fbfm.configuration_rtc import RTCConfig
from fbfm.policies.fbfm.debug_tracker import Tracker

logger = logging.getLogger(__name__)


class PrevChunk:
    constrain_mode = "Feedback"
    actions: Tensor | None = None
    states: Tensor | None = None
    inference_delay: int = 0
    action_constrained_num: int = 0
    state_constrained_num: int = 0

    def __init__(
        self,
        constrain_mode: str = "Feedback",
        actions: Tensor | None = None,
        action_constrained_num: int = 0,
        action_num: int = 16,
        action_dim: int = 16,
        states: Tensor | None = None,
        state_constrained_num: int = 0,
        state_num: int = 4,
        state_dim: int = 128,
        inference_delay: int = 0,
    ):
        self.constrain_mode = constrain_mode
        
        # 动作相关配置 - 始终初始化为固定尺寸
        self.action_num = action_num
        self.action_dim = action_dim
        self.action_constrained_num = 0
        self.actions = torch.zeros(action_num, action_dim)  # 固定尺寸 (T_a, D_a)
        
        if actions is not None:
            actual_constrained = min(action_constrained_num, action_num, actions.shape[0])
            self.actions[:actual_constrained] = actions[:actual_constrained].clone()
            self.action_constrained_num = actual_constrained
        else:
            self.action_constrained_num = min(action_constrained_num, action_num)

        # 状态相关配置 - 始终初始化为固定尺寸
        self.state_num = state_num
        self.state_dim = state_dim
        self.states = torch.zeros(state_num, state_dim)  # 固定尺寸 (T_s, D_s)
        self.state_constrained_num = 0  # 初始为 0
        
        # 如果提供了初始状态，则设置
        if states is not None and states.shape[1] == state_dim:
            actual_constrained = min(state_constrained_num, state_num, states.shape[0])
            self.states[:actual_constrained] = states[:actual_constrained].clone()
            self.state_constrained_num = actual_constrained
        
        self.inference_delay = inference_delay

    def append_new_state(self, new_state: Tensor) -> None:
        """Append a newly observed state by replacing at state_constrained_num index.

        This maintains a fixed-size states tensor of shape (state_num, state_dim).
        """
        # Normalize to 2D (1, D)
        if new_state.dim() == 1:
            new_state = new_state.unsqueeze(0)
        elif new_state.dim() == 2:
            pass
        else:
            raise ValueError(
                f"append_new_state expects a 1D or 2D tensor, got shape {tuple(new_state.shape)}"
            )
        
        if new_state.shape[1] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: expected {self.state_dim}, got {new_state.shape[1]}"
            )
            
        # 替换当前 state_constrained_num 位置的状态（如果未满）
        if self.state_constrained_num < self.state_num:
            self.states[self.state_constrained_num] = new_state[0]
            self.state_constrained_num += 1
        # 如果已满，可以选择覆盖最老状态或丢弃新状态
        # 这里选择丢弃，保持已填充状态的顺序性

    def get_action_prefix_weights(self) -> Tensor:
        """Get the weights for the action prefix.

        This is intended to be used as a guide for the next chunk generation.
        """
        if self.constrain_mode == "None":
            action_prefix_weights = torch.zeros(self.action_num)
        elif self.constrain_mode == "RTC" or self.constrain_mode == "Feedback":
            action_prefix_weights = torch.zeros(self.action_num)
            action_prefix_weights[:(self.action_constrained_num)] = 1
        else:
            raise ValueError(f"Unknown constrain mode: {self.constrain_mode}")
        return action_prefix_weights
    
    def get_state_prefix_weights(self) -> Tensor:
        """Get the weights for the state prefix.

        This is intended to be used as a guide for the next chunk generation.
        """
        if self.constrain_mode == "Feedback":
            state_prefix_weights = torch.zeros(self.state_num)
            state_prefix_weights[:(self.state_constrained_num)] = 1
        elif self.constrain_mode == "RTC" or self.constrain_mode == "None":
            state_prefix_weights = torch.zeros(self.state_num)
        else:
            raise ValueError(f"Unknown constrain mode: {self.constrain_mode}")
        return state_prefix_weights

    def get_prefix_weights(self) -> Tensor:
        """Get the weights for the chunk prefix.

        This is intended to be used as a guide for the next chunk generation.
        
        Returns:
            Tensor of shape (state_num + action_num,) where:
            - First action_num elements: first action_constrained_num are 1, rest 0
            - Last state_num elements: first state_constrained_num are 1, rest 0
        """
        action_prefix_weights = self.get_action_prefix_weights()
        state_prefix_weights = self.get_state_prefix_weights()
        # Concatenate: action part first, then state part
        chunk_prefix_weights = torch.cat([action_prefix_weights, state_prefix_weights])
        return chunk_prefix_weights
    
    def get_constrain_mode(self) -> str:
        """Get the constrain mode for the chunk."""
        return self.constrain_mode
    
    def get_constrained_states(self) -> Tensor:
        """Get the constrained states for the chunk."""
        return self.states
    
    def get_constrained_actions(self) -> Tensor:
        """Get the constrained actions for the chunk."""
        return self.actions


@dataclass
class RTCPrevChunk:
    """Container for previous chunk leftovers used by RTC.

    This structure is intended for the extended RTC variant where both:
    - action leftovers (from the previous action chunk), and
    - observed state latents (VAE-encoded images during execution)

    are passed together to guide the next chunk generation.

    It does not change the existing RTC behavior: callers may keep passing
    a plain Tensor as ``prev_chunk_left_over`` to ``denoise_step``.  This
    class is a typed way to bundle future state+action information once the
    call sites are updated.
    """

    action: Tensor | None = None
    """Unexecuted actions from the previous chunk."""

    state: Tensor | None = None
    """Observed state latents (e.g. VAE-encoded image features) during execution."""

    inference_delay: int = 0
    """Number of timesteps of inference delay associated with this leftover."""

    execution_horizon: int | None = None
    """Optional horizon for action prefix weights (falls back to config if None)."""

    state_observed_horizon: int | None = None
    """Optional horizon for state prefix weights (falls back to execution_horizon if None)."""

    def append_state_latent(self, new_latent: Tensor) -> None:
        """Append a newly observed state latent along the time dimension.

        This is intended to be called incrementally while executing a chunk:
        each time a new image is observed and encoded by the VAE encoder, the
        resulting latent can be appended here. Internally we store state
        latents as a 2D tensor of shape ``(T, state_dim)`` where ``T`` is the
        number of collected steps (no batch dimension).

        Args:
            new_latent: Single-step state latent with shape ``(state_dim,)`` or
                ``(1, state_dim)``. A small sequence ``(T_new, state_dim)`` is
                also accepted and concatenated in order.
        """
        if new_latent is None:
            return

        if not isinstance(new_latent, Tensor):
            raise TypeError(f"new_latent must be a Tensor, got {type(new_latent)}")

        # Normalize to 2D (T_new, D)
        if new_latent.dim() == 1:
            new_latent = new_latent.unsqueeze(0)
        elif new_latent.dim() == 2:
            pass
        else:
            raise ValueError(
                f"append_state_latent expects a 1D or 2D tensor, got shape {tuple(new_latent.shape)}"
            )

        if self.state is None:
            # First observation
            self.state = new_latent
            return

        if self.state.dim() != 2:
            raise ValueError(
                f"RTCPrevChunk.state is expected to be 2D (T, D), got shape {tuple(self.state.shape)}"
            )

        if self.state.shape[1] != new_latent.shape[1]:
            raise ValueError(
                "State latent dimension mismatch when appending: "
                f"existing D={self.state.shape[1]}, new D={new_latent.shape[1]}"
            )

        self.state = torch.cat([self.state, new_latent], dim=0)
        self.state_observed_horizon = self.state_observed_horizon + 1 # 维护接收到的状态的长度
