#!/usr/bin/env python

import copy
import logging
import threading
from collections import deque

import torch
import torch.nn as nn

from fbfm.policies.fbfm.modeling_fbfm_rtc import (
    RTCPrevChunk,
    RTCProcessor,
    prepare_prev_chunk_left_over,
)
from fbfm.policies.fbfm.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class FBFMPolicy(nn.Module):
    def __init__(
        self,
        config: RTCConfig,
        model: torch.nn.Module,
        vae_encoder,
        initial_action_chunk: torch.Tensor,
        s_chunk: int = 5,
        s_step: int = 1,
        delay_buffer_size: int = 5,
        initial_delay: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.vae_encoder = vae_encoder
        self.s_chunk = s_chunk
        self.s_step = s_step
        self.initial_delay = initial_delay

        self.processor = RTCProcessor(config)

        self.t = 0
        self.A_cur = initial_action_chunk
        self.o_cur = None
        self.PC = RTCPrevChunk()
        self.PC.action = initial_action_chunk
        self.Q = deque([self.initial_delay], maxlen=delay_buffer_size)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.running = True

        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
        )
        self.inference_thread.start()

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the next action for the given observation."""
        with self.lock:
            self.t += 1
            self.o_cur = obs
            logger.debug("Updated observation for step %s", self.t)
            self.cond.notify_all()

        z_cur = self.vae_encoder.encode(obs)
        logger.debug("Encoded observation to latent with shape %s", getattr(z_cur, "shape", None))
        self.PC.append_state_latent(z_cur)

        with self.lock:
            while self.A_cur is None:
                logger.debug("Waiting for action chunk to be ready")
                self.cond.wait()

            action = self.A_cur[self.t - 1].clone()
            logger.debug("Selected action for step %s", self.t)
            return action

    def _inference_loop(self) -> None:
        """Run asynchronous model inference in a background thread."""
        logger.debug("Inference loop started")
        while self.running:
            with self.lock:
                while self.t < self.config.s_chunk and self.running:
                    self.cond.wait()

                if not self.running:
                    break

                s = self.t
                o = copy.deepcopy(self.o_cur)
                remaining_actions = self.A_cur[s:].clone().detach()
                d = max(self.Q) if len(self.Q) > 0 else 0
                prev_chunk = prepare_prev_chunk_left_over(
                    action_left_over=remaining_actions,
                    observed_state_latents=self.PC.state.clone() if self.PC.state is not None else None,
                    inference_delay=d,
                    execution_horizon=self.config.execution_horizon,
                    state_execution_horizon=getattr(self.config, "state_execution_horizon", None),
                )

            logger.debug("Launching guided inference with delay %s", d)
            A_new = self._guided_inference(o, prev_chunk, s)

            with self.lock:
                self.A_cur = A_new
                self.t = 0
                self.PC = RTCPrevChunk()
                self.PC.action = A_new.clone()
                self.Q.append(s)
                self.cond.notify_all()

        logger.debug("Inference loop exiting")

    def _guided_inference(
        self,
        obs: torch.Tensor,
        prev_chunk: RTCPrevChunk | None,
        executed_steps: int,
    ) -> torch.Tensor:
        """Perform a guided diffusion inference to update action chunks."""
        H = self.config.chunk_size
        state_dim = self.config.chunk_state_dim
        action_dim = self.config.chunk_action_dim
        n_steps = self.config.num_inference_steps
        noise_scale = getattr(self.config, "noise_scale", 1.0)

        device = next(self.model.parameters()).device
        X = torch.randn(H, state_dim + action_dim, device=device) * noise_scale
        X = X.requires_grad_(True)

        logger.debug(
            "Starting guided inference: H=%s state_dim=%s action_dim=%s steps=%s",
            H,
            state_dim,
            action_dim,
            n_steps,
        )

        for i in range(n_steps):
            tau = i / n_steps
            time = 1 - tau

            def orig_denoise(x):
                return self.model(x, obs, tau)

            v_guided = self.processor.denoise_step(
                x_t=X,
                prev_chunk_left_over=prev_chunk,
                inference_delay=prev_chunk.inference_delay if prev_chunk else 0,
                time=time,
                original_denoise_step_partial=orig_denoise,
                execution_horizon=self.config.execution_horizon,
            )

            X = X + (1.0 / n_steps) * v_guided
            X = X.detach().requires_grad_(True)

        logger.debug("Guided inference completed")
        return X[:, state_dim:].detach()

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        pass

    def close(self) -> None:
        """Stop the background thread and release resources."""
        pass
