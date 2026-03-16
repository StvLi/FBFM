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
Real Time Chunking (RTC) and Bidirectional Decoding (BID) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from fbfm.configs.types import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = False

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # State feedback (state = VAE-encoded image latent, not joint angles)
    state_feedback_enabled: bool = False
    state_observed_horizon: int = 10
    state_max_guidance_weight: float = 10.0
    """Fixed state dimension per timestep in the chunk (not tied to VAE latent dim). Required when state_feedback_enabled."""
    chunk_state_dim: int | None = None
    """Fixed action dimension per timestep in the chunk. Required when state_feedback_enabled. Must satisfy chunk_state_dim + chunk_action_dim == x_t last dim."""
    chunk_action_dim: int | None = None
    """Dimension of observed state latent (e.g. VAE output). Used to validate appended state; typically equals chunk_state_dim."""
    state_latent_dim: int | None = None

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
        if self.state_feedback_enabled:
            if self.state_max_guidance_weight <= 0:
                raise ValueError(
                    f"state_max_guidance_weight must be positive when state_feedback_enabled, "
                    f"got {self.state_max_guidance_weight}"
                )
            if self.chunk_state_dim is None or self.chunk_state_dim <= 0:
                raise ValueError(
                    "chunk_state_dim must be set and positive when state_feedback_enabled"
                )
            if self.chunk_action_dim is None or self.chunk_action_dim <= 0:
                raise ValueError(
                    "chunk_action_dim must be set and positive when state_feedback_enabled"
                )
