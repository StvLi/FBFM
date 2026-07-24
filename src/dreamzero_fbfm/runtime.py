"""Runtime hook that inserts joint FBFM at DreamZero DiT evaluations."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from .audit import JsonlAudit
from .constraints import ActionNormalizer, ChunkConstraints, ConstraintMode
from .guidance import joint_fbfm_guidance
from .pseudo_clock import SolverClock


class InferenceCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class FeedbackObservation:
    action_offset: int
    main_image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    task_description: str


def flow_sigmas(num_steps: int, shift: float, sigma_max: float = 0.999) -> tuple[float, ...]:
    base = np.linspace(sigma_max, 0.0, num_steps + 1, dtype=np.float64)[:-1]
    shifted = shift * base / (1 + (shift - 1) * base)
    return tuple(float(value) for value in shifted)


class DreamZeroFeedbackEncoder:
    """Encode four aligned LIBERO observations into one native DreamZero latent."""

    def __init__(self, policy: Any, *, stride: int = 2, observations_per_latent: int = 4) -> None:
        if stride <= 0 or observations_per_latent <= 0:
            raise ValueError("feedback encoder cadence must be positive")
        self.policy = policy
        self.head = policy.action_head
        self.stride = stride
        self.observations_per_latent = observations_per_latent
        self.reset()

    def reset(self) -> None:
        self._frames: list[Tensor] = []

    def add(self, feedback: FeedbackObservation) -> list[Tensor]:
        if feedback.action_offset <= 0 or feedback.action_offset % self.stride:
            return []
        env_obs = {
            "main_images": np.asarray(feedback.main_image, dtype=np.uint8)[None],
            "wrist_images": np.asarray(feedback.wrist_image, dtype=np.uint8)[None],
            "states": np.asarray(feedback.state, dtype=np.float32)[None],
            "task_descriptions": [feedback.task_description],
        }
        from tianshou.data import Batch

        converted = self.policy._observation_convert(env_obs)
        normalized = self.policy._process_batch(Batch(obs=converted))
        frame = normalized["images"]
        if tuple(frame.shape[:2]) != (1, 1):
            raise ValueError(f"Unexpected transformed feedback shape {tuple(frame.shape)}")
        self._frames.append(frame.detach().cpu())
        encoded: list[Tensor] = []
        while len(self._frames) >= self.observations_per_latent:
            group = self._frames[: self.observations_per_latent]
            del self._frames[: self.observations_per_latent]
            encoded.append(self._encode(torch.cat(group, dim=1)))
        return encoded

    def _encode(self, images: Tensor) -> Tensor:
        device = next(self.head.parameters()).device
        dtype = next(self.head.parameters()).dtype
        video = images.to(device=device)
        video = video.permute(0, 4, 1, 2, 3).float() / 255.0
        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        flat = self.head.normalize_video(flat)
        target_size = (
            int(self.head.config.target_video_height),
            int(self.head.config.target_video_width),
        )
        flat = torch.nn.functional.interpolate(
            flat, size=target_size, mode="bilinear", align_corners=False
        )
        video = flat.reshape(batch, frames, channels, *target_size).permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            latent = self.head.vae.encode(
                video.to(dtype=dtype),
                tiled=self.head.tiled,
                tile_size=(self.head.tile_size_height, self.head.tile_size_width),
                tile_stride=(self.head.tile_stride_height, self.head.tile_stride_width),
            ).detach()
        if latent.ndim != 5 or latent.shape[2] < 1:
            raise ValueError(f"Unexpected feedback latent shape {tuple(latent.shape)}")
        return latent[:, :, -1]


class DreamZeroFBFMRuntime:
    """Own one active pseudo-asynchronous chunk and its DreamZero solver hook."""

    def __init__(
        self,
        policy: Any,
        normalizer: ActionNormalizer,
        *,
        mode: ConstraintMode | str,
        delay: int = 8,
        execution_horizon: int = 8,
        solver_steps: int = 16,
        sigma_shift: float = 5.0,
        beta: float = 10.0,
        audit_path: str | None = None,
    ) -> None:
        if delay != execution_horizon:
            raise ValueError("This matched DreamZero protocol requires d == s")
        if delay <= 0 or delay + execution_horizon != policy.action_head.action_horizon:
            raise ValueError("DreamZero overlap protocol requires d+s == H")
        self.policy = policy
        self.head = policy.action_head
        self.normalizer = normalizer
        self.mode = ConstraintMode.parse(mode)
        self.delay = delay
        self.execution_horizon = execution_horizon
        self.solver_steps = solver_steps
        self.beta = beta
        self.sigmas = flow_sigmas(solver_steps, sigma_shift)
        self.audit = JsonlAudit(audit_path)
        self.clock = SolverClock()
        self.feedback_encoder = DreamZeroFeedbackEncoder(policy)
        self._feedback: queue.Queue[FeedbackObservation] = queue.Queue()
        self._constraints: ChunkConstraints | None = None
        self._action_targets: Tensor | None = None
        self._action_mask: Tensor | None = None
        self._step = 0
        self._next_state_slot = 0
        self._lock = threading.RLock()
        self._install_hook()
        # FBFM requires an endpoint graph at each solver evaluation. Keep this
        # identical for NONE/RTC/FBFM within the matched protocol.
        self.head.dit_step_mask = [True] * self.solver_steps

    def begin_chunk(self, committed_actions: np.ndarray | None, *, pseudo_async: bool) -> None:
        with self._lock:
            self.cancel()
            self._step = 0
            self._next_state_slot = 0
            self._constraints = None
            self.feedback_encoder.reset()
            self._feedback = queue.Queue()
            horizon = int(self.head.action_horizon)
            model_dim = int(self.head.model.action_dim)
            target = torch.zeros(1, horizon, model_dim, dtype=torch.float32)
            mask = torch.zeros_like(target)
            if committed_actions is not None:
                physical = np.asarray(committed_actions, dtype=np.float32)
                if physical.shape != (self.delay, self.normalizer.environment_dim):
                    raise ValueError(
                        f"committed actions must be {(self.delay, self.normalizer.environment_dim)}, "
                        f"got {physical.shape}"
                    )
                normalized = self.normalizer.normalize(physical)
                target[0, : self.delay] = normalized
                mask[0, : self.delay, : self.normalizer.environment_dim] = 1
            self._action_targets = target
            self._action_mask = mask
            self.clock.start(enabled=pseudo_async)
            self.audit.write(
                "chunk_begin",
                mode=self.mode.value,
                pseudo_async=pseudo_async,
                action_mask_nonzero=int(torch.count_nonzero(mask).item()),
            )

    def submit_feedback(self, feedback: FeedbackObservation) -> None:
        self._feedback.put(feedback)

    def cancel(self) -> None:
        self.clock.close()
        if self._constraints is not None:
            self._constraints.close()

    def _ensure_constraints(self, video: Tensor, action: Tensor) -> ChunkConstraints:
        if self._constraints is None:
            if self._action_targets is None or self._action_mask is None:
                raise RuntimeError("begin_chunk must be called before inference")
            self._constraints = ChunkConstraints(
                mode=self.mode,
                action_targets=self._action_targets.to(device=action.device, dtype=action.dtype),
                action_mask=self._action_mask.to(device=action.device, dtype=action.dtype),
                state_targets=torch.zeros_like(video),
                state_mask=torch.zeros(
                    video.shape[0], 1, video.shape[2], 1, 1,
                    device=video.device, dtype=video.dtype,
                ),
            )
        return self._constraints

    def _drain_feedback(self, constraints: ChunkConstraints) -> int:
        drained = 0
        while True:
            try:
                item = self._feedback.get_nowait()
            except queue.Empty:
                break
            drained += 1
            for latent in self.feedback_encoder.add(item):
                accepted = constraints.update_state_slot(self._next_state_slot, latent)
                if accepted:
                    self._next_state_slot += 1
        return drained

    def _install_hook(self) -> None:
        original: Callable[..., Any] = self.head._run_diffusion_steps

        def hooked_run_diffusion_steps(**kwargs: Any) -> Any:
            is_solver = (
                kwargs.get("action") is not None
                and not kwargs.get("kv_cache_metadata", {}).get("update_kv_cache", False)
            )
            if not is_solver:
                with torch.no_grad():
                    return original(**kwargs)
            if self._step >= self.solver_steps:
                raise RuntimeError("DreamZero issued more solver evaluations than configured")
            if not self.clock.consume():
                raise InferenceCancelled("pseudo-asynchronous chunk cancelled")

            video = kwargs["noisy_input"].detach().clone().requires_grad_(True)
            action = kwargs["action"].detach().clone().requires_grad_(True)
            constraints = self._ensure_constraints(video, action)
            drained = self._drain_feedback(constraints)
            state_target, state_mask, action_target, action_mask, version = constraints.snapshot()
            has_guidance = bool(torch.any(state_mask).item() or torch.any(action_mask).item())
            call_kwargs = dict(kwargs, noisy_input=video, action=action)

            if has_guidance:
                with torch.enable_grad():
                    predictions = original(**call_kwargs)
                    conditional_video, conditional_action = predictions[0]
                    unconditional_video, unconditional_action = predictions[1]
                    base_video = unconditional_video + self.head.cfg_scale * (
                        conditional_video - unconditional_video
                    )
                    result = joint_fbfm_guidance(
                        video_sample=video,
                        action_sample=action,
                        video_velocity=base_video,
                        action_velocity=conditional_action,
                        video_target=state_target,
                        video_mask=state_mask,
                        action_target=action_target,
                        action_mask=action_mask,
                        sigma=self.sigmas[self._step],
                        beta=self.beta,
                    )
                # The upstream caller applies CFG after this hook. Equal video
                # branches make that operation an identity for the guided field.
                predictions = [
                    (result.video_velocity, result.action_velocity),
                    (result.video_velocity, unconditional_action.detach()),
                ]
                diagnostics = result.diagnostics
            else:
                with torch.no_grad():
                    predictions = original(**kwargs)
                diagnostics = {
                    "guided": False,
                    "state_mask_nonzero": 0,
                    "action_mask_nonzero": 0,
                    "state_error_norm": 0.0,
                    "action_error_norm": 0.0,
                    "video_correction_norm": 0.0,
                    "action_correction_norm": 0.0,
                    "guidance_weight": 0.0,
                }

            self.audit.write(
                "solver_step",
                mode=self.mode.value,
                step=self._step,
                sigma=self.sigmas[self._step],
                context_version=version,
                feedback_drained=drained,
                gpu_allocated_bytes=(
                    torch.cuda.memory_allocated(video.device) if video.is_cuda else 0
                ),
                **diagnostics,
            )
            self._step += 1
            self.clock.complete()
            return predictions

        self.head._run_diffusion_steps = hooked_run_diffusion_steps
