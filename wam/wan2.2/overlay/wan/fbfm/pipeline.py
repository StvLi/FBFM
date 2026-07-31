"""Wan2.2 TI2V inference with LingBot-style visual state feedback."""

from __future__ import annotations

import gc
import math
import random
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from wan.textimage2video import WanTI2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import best_output_size, masks_like

from .feedback import (
    DEFAULT_KP,
    DEFAULT_STATE_WEIGHT,
    FeedbackEvent,
    FeedbackMode,
    NativeWanStreamingEncoder,
    StateFeedbackController,
    endpoint_state_guidance_from_cfg_vjps,
    masked_endpoint_error,
)


@dataclass(frozen=True)
class WanFBFMOutput:
    video: torch.Tensor | None
    audit: list[dict[str, Any]]
    seed: int
    output_size: tuple[int, int]


def resize_center_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    """Apply exactly the native Wan TI2V I2V spatial transform."""
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized = image.resize(
        (round(source_width * scale), round(source_height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def image_to_video_tensor(
    image: Image.Image, width: int, height: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Convert one RGB frame to normalized (C,1,H,W) Wan video layout."""
    frame = resize_center_crop(image.convert("RGB"), width, height)
    value = TF.to_tensor(frame).sub_(0.5).div_(0.5).unsqueeze(1)
    return value.to(device) if device is not None else value


def build_feedback_events(
    *,
    future_frames: Sequence[Image.Image],
    release_steps: Sequence[int],
    width: int,
    height: int,
    observations_per_slot: int = 4,
) -> list[FeedbackEvent]:
    """Map consecutive measured frames to future latent slots without padding."""
    required = len(release_steps) * observations_per_slot
    if len(future_frames) < required:
        raise ValueError(
            f"feedback schedule needs {required} future frames, got {len(future_frames)}"
        )
    events = []
    for event_index, release_step in enumerate(release_steps):
        start = event_index * observations_per_slot
        chunk = torch.cat(
            [
                image_to_video_tensor(frame, width, height)
                for frame in future_frames[start : start + observations_per_slot]
            ],
            dim=1,
        )
        events.append(
            FeedbackEvent(
                release_step=int(release_step),
                slot=event_index + 1,
                frames=chunk,
            )
        )
    return events


class WanTI2VFBFM(WanTI2V):
    """State-only FBFM variant of the official Wan2.2 TI2V pipeline."""

    def generate_with_feedback(
        self,
        *,
        input_prompt: str,
        image: Image.Image,
        mode: FeedbackMode | str,
        feedback_frames: Sequence[Image.Image] = (),
        feedback_release_steps: Sequence[int] = (),
        max_area: int = 704 * 1280,
        frame_num: int = 121,
        shift: float = 5.0,
        sampling_steps: int = 50,
        guide_scale: float = 5.0,
        n_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        beta: float = 10.0,
        state_weight: float = DEFAULT_STATE_WEIGHT,
        kp: float = DEFAULT_KP,
        gradient_checkpointing: bool = True,
    ) -> WanFBFMOutput:
        """Generate one paired DIRECT/FBFM video prediction.

        ``feedback_frames`` contains measured future frames after ``image``.
        Every four consecutive frames form one causal Wan2.2 latent slot. The
        corresponding entry in ``feedback_release_steps`` selects the solver
        boundary at which that raw observation group becomes visible.
        """
        feedback_mode = FeedbackMode.parse(mode)
        if frame_num <= 1 or (frame_num - 1) % self.vae_stride[0]:
            raise ValueError(f"frame_num must be 4n+1 for this VAE, got {frame_num}")
        if sampling_steps <= 1:
            raise ValueError("sampling_steps must be greater than one")
        if any(step < 0 or step >= sampling_steps for step in feedback_release_steps):
            raise ValueError("feedback release steps must fall inside the solver loop")
        if list(feedback_release_steps) != sorted(feedback_release_steps):
            raise ValueError("feedback release steps must be nondecreasing")
        if feedback_mode is FeedbackMode.FBFM and not feedback_release_steps:
            raise ValueError("FBFM mode requires at least one feedback release step")

        source_width, source_height = image.size
        spatial_stride_width = self.patch_size[2] * self.vae_stride[2]
        spatial_stride_height = self.patch_size[1] * self.vae_stride[1]
        output_width, output_height = best_output_size(
            source_width,
            source_height,
            spatial_stride_width,
            spatial_stride_height,
            max_area,
        )
        anchor = image_to_video_tensor(
            image,
            output_width,
            output_height,
            device=self.device,
        )

        latent_frame_count = (frame_num - 1) // self.vae_stride[0] + 1
        sequence_length = (
            latent_frame_count
            * (output_height // self.vae_stride[1])
            * (output_width // self.vae_stride[2])
            // (self.patch_size[1] * self.patch_size[2])
        )
        sequence_length = math.ceil(sequence_length / self.sp_size) * self.sp_size

        resolved_seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(resolved_seed)
        noise = torch.randn(
            self.vae.model.z_dim,
            latent_frame_count,
            output_height // self.vae_stride[1],
            output_width // self.vae_stride[2],
            dtype=torch.float32,
            generator=generator,
            device=self.device,
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [value.to(self.device) for value in context]
            context_null = [value.to(self.device) for value in context_null]

        anchor_latent = self.vae.encode([anchor])
        if anchor_latent is None:
            raise RuntimeError("Wan VAE failed to encode the I2V anchor")
        mask_primary, mask_noise = masks_like([noise], zero=True)
        del mask_primary
        latent = (1.0 - mask_noise[0]) * anchor_latent[0] + mask_noise[0] * noise

        events = build_feedback_events(
            future_frames=feedback_frames,
            release_steps=feedback_release_steps,
            width=output_width,
            height=output_height,
            observations_per_slot=NativeWanStreamingEncoder.observations_per_slot,
        )
        controller = StateFeedbackController(
            mode=feedback_mode,
            encoder=NativeWanStreamingEncoder(self.vae),
            anchor=anchor,
            latent_template=noise,
            events=events,
            beta=beta,
            state_weight=state_weight,
            kp=kp,
        )
        controller.audit.insert(
            0,
            {
                "event": "generation_begin",
                "mode": feedback_mode.value,
                "seed": resolved_seed,
                "frame_num": frame_num,
                "latent_frame_count": latent_frame_count,
                "output_width": output_width,
                "output_height": output_height,
                "sampling_steps": sampling_steps,
                "feedback_release_steps": list(feedback_release_steps),
                "beta": beta,
                "state_weight": state_weight,
                "kp": kp,
                "gradient_checkpointing": gradient_checkpointing,
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
        )

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(
            sampling_steps,
            device=self.device,
            shift=shift,
        )

        conditional_args = {
            "context": [context[0]],
            "seq_len": sequence_length,
        }
        unconditional_args = {
            "context": context_null,
            "seq_len": sequence_length,
        }

        if offload_model or self.init_on_cpu:
            self.model.to(self.device)
            torch.cuda.empty_cache()
        self.model.gradient_checkpointing = gradient_checkpointing
        torch.cuda.reset_peak_memory_stats(self.device)

        with torch.amp.autocast("cuda", dtype=self.param_dtype), no_sync():
            for solver_step, timestep_value in enumerate(tqdm(scheduler.timesteps)):
                feedback_updates = controller.advance(solver_step)
                timestep = torch.stack([timestep_value]).to(self.device)
                temporal_timestep = (mask_noise[0][0][:, ::2, ::2] * timestep).flatten()
                temporal_timestep = torch.cat(
                    [
                        temporal_timestep,
                        temporal_timestep.new_ones(
                            sequence_length - temporal_timestep.size(0)
                        )
                        * timestep,
                    ]
                ).unsqueeze(0)

                target, active_mask, constraint_version = (
                    controller.constraints.snapshot(feedback_mode)
                )
                has_feedback = bool(torch.any(active_mask).item())
                if has_feedback:
                    sample = latent.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        conditional_velocity = self.model(
                            [sample], t=temporal_timestep, **conditional_args
                        )[0]
                    with torch.no_grad():
                        unconditional_velocity = self.model(
                            [sample], t=temporal_timestep, **unconditional_args
                        )[0]
                        native_velocity = unconditional_velocity + guide_scale * (
                            conditional_velocity.detach() - unconditional_velocity
                        )
                    sigma = scheduler.sigmas[solver_step]
                    error = masked_endpoint_error(
                        sample=sample,
                        velocity=native_velocity,
                        target=target,
                        mask=active_mask,
                        sigma=sigma,
                        state_weight=state_weight,
                    )
                    conditional_vjp = torch.autograd.grad(
                        conditional_velocity,
                        sample,
                        grad_outputs=error,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    del conditional_velocity
                    with torch.enable_grad():
                        unconditional_for_vjp = self.model(
                            [sample], t=temporal_timestep, **unconditional_args
                        )[0]
                    unconditional_vjp = torch.autograd.grad(
                        unconditional_for_vjp,
                        sample,
                        grad_outputs=error,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    guidance = endpoint_state_guidance_from_cfg_vjps(
                        velocity=native_velocity,
                        error=error,
                        mask=active_mask,
                        conditional_vjp=conditional_vjp,
                        unconditional_vjp=unconditional_vjp,
                        sigma=sigma,
                        guide_scale=guide_scale,
                        beta=beta,
                        kp=kp,
                    )
                    controller.audit.append(
                        {
                            "event": "solver_guidance",
                            "solver_step": solver_step,
                            "sigma": float(sigma),
                            "constraint_version": constraint_version,
                            "feedback_updates": feedback_updates,
                            **guidance.diagnostics,
                        }
                    )
                    velocity = guidance.velocity
                    del (
                        sample,
                        unconditional_velocity,
                        unconditional_for_vjp,
                        native_velocity,
                        error,
                        conditional_vjp,
                        unconditional_vjp,
                    )
                else:
                    with torch.no_grad():
                        conditional_velocity = self.model(
                            [latent], t=temporal_timestep, **conditional_args
                        )[0]
                        unconditional_velocity = self.model(
                            [latent], t=temporal_timestep, **unconditional_args
                        )[0]
                        velocity = unconditional_velocity + guide_scale * (
                            conditional_velocity - unconditional_velocity
                        )
                    controller.audit.append(
                        {
                            "event": "solver_guidance",
                            "solver_step": solver_step,
                            "sigma": float(scheduler.sigmas[solver_step]),
                            "constraint_version": constraint_version,
                            "feedback_updates": feedback_updates,
                            "guided": False,
                            "mask_nonzero": 0,
                        }
                    )
                    del conditional_velocity, unconditional_velocity

                with torch.no_grad():
                    next_latent = scheduler.step(
                        velocity.unsqueeze(0),
                        timestep_value,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0]
                    latent = next_latent.squeeze(0).detach()
                    latent = (1.0 - mask_noise[0]) * anchor_latent[0] + mask_noise[
                        0
                    ] * latent
                del velocity, temporal_timestep

        if not torch.isfinite(latent).all():
            raise FloatingPointError("Wan2.2 produced a non-finite video latent")
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)
        controller.constraints.close()
        if offload_model:
            self.model.cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        video = self.vae.decode([latent])[0] if self.rank == 0 else None
        controller.audit.append(
            {
                "event": "generation_end",
                "constraint_version": controller.constraints.version,
                "peak_cuda_memory_allocated_bytes": peak_allocated,
                "peak_cuda_memory_reserved_bytes": peak_reserved,
            }
        )

        del noise, latent, scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return WanFBFMOutput(
            video=video,
            audit=controller.audit,
            seed=resolved_seed,
            output_size=(output_width, output_height),
        )
