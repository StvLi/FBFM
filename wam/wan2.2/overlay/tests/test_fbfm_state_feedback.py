from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from wan.fbfm.feedback import (
    DEFAULT_KP,
    DEFAULT_STATE_WEIGHT,
    FeedbackEvent,
    FeedbackMode,
    StateFeedbackController,
    endpoint_state_guidance,
    endpoint_state_guidance_from_cfg_vjps,
    evenly_spaced_release_steps,
    masked_endpoint_error,
)
from wan.fbfm.pipeline import build_feedback_events
from wan.modules import attention as attention_module
from wan.modules.model import WanModel


class FakeFeedbackEncoder:
    observations_per_slot = 4

    def __init__(self) -> None:
        self.primed = False

    def prime(self, anchor: torch.Tensor) -> torch.Tensor:
        self.primed = True
        return anchor

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if not self.primed:
            raise RuntimeError("not primed")
        return frames.mean(dim=1, keepdim=True)


def test_visual_only_default_state_weight_is_unity():
    assert DEFAULT_STATE_WEIGHT == 1.0
    assert DEFAULT_KP == 1.0


def test_controller_releases_feedback_only_at_solver_boundary():
    template = torch.zeros(1, 3, 1, 1)
    controller = StateFeedbackController(
        mode=FeedbackMode.FBFM,
        encoder=FakeFeedbackEncoder(),
        anchor=torch.zeros(1, 1, 1, 1),
        latent_template=template,
        events=[
            FeedbackEvent(
                release_step=2,
                slot=1,
                frames=torch.full((1, 4, 1, 1), 3.0),
            )
        ],
    )

    assert controller.advance(1) == 0
    _, mask_before, version_before = controller.constraints.snapshot("FBFM")
    assert torch.count_nonzero(mask_before).item() == 0
    assert version_before == 0

    assert controller.advance(2) == 1
    target, mask_after, version_after = controller.constraints.snapshot("FBFM")
    assert mask_after.flatten().tolist() == [0.0, 1.0, 0.0]
    assert target[:, 1].item() == pytest.approx(3.0)
    assert version_after == 1


def test_direct_mode_keeps_state_mask_zero():
    controller = StateFeedbackController(
        mode="DIRECT",
        encoder=FakeFeedbackEncoder(),
        anchor=torch.zeros(1, 1, 1, 1),
        latent_template=torch.zeros(1, 2, 1, 1),
        events=[],
    )
    _, mask, version = controller.constraints.snapshot("DIRECT")
    assert torch.count_nonzero(mask).item() == 0
    assert version == 0


def test_endpoint_guidance_moves_an_euler_update_toward_target():
    sample = torch.tensor([[[[2.0]]]], requires_grad=True)
    velocity = sample * 0
    result = endpoint_state_guidance(
        sample=sample,
        velocity=velocity,
        target=torch.zeros_like(sample),
        mask=torch.ones(1, 1, 1, 1),
        sigma=0.5,
        beta=1.0,
    )
    next_sample = sample.detach() + result.velocity * (0.0 - 0.5)

    assert result.diagnostics["guided"] is True
    assert next_sample.abs().item() < sample.detach().abs().item()


def test_kp_scales_complete_feedback_velocity_correction():
    def guide(kp: float):
        sample = torch.tensor([[[[2.0]]]], requires_grad=True)
        return endpoint_state_guidance(
            sample=sample,
            velocity=sample * 0,
            target=torch.zeros_like(sample),
            mask=torch.ones(1, 1, 1, 1),
            sigma=0.5,
            beta=1.0,
            kp=kp,
        )

    full = guide(1.0)
    damped = guide(0.05)

    torch.testing.assert_close(damped.velocity, full.velocity * 0.05)
    assert damped.diagnostics["kp"] == pytest.approx(0.05)


def test_split_cfg_vjps_match_combined_endpoint_guidance():
    sample = torch.randn(2, 3, 2, 2, requires_grad=True)
    target = torch.randn_like(sample)
    mask = torch.tensor([0.0, 1.0, 1.0]).view(1, 3, 1, 1)
    sigma = 0.6
    guide_scale = 5.0

    conditional = sample.sin() + sample.square()
    unconditional = sample.cos() - 0.25 * sample
    velocity = unconditional + guide_scale * (conditional - unconditional)
    combined = endpoint_state_guidance(
        sample=sample,
        velocity=velocity,
        target=target,
        mask=mask,
        sigma=sigma,
        beta=3.0,
        state_weight=0.2,
        kp=0.05,
    )

    error = masked_endpoint_error(
        sample=sample,
        velocity=velocity,
        target=target,
        mask=mask,
        sigma=sigma,
        state_weight=0.2,
    )
    conditional = sample.sin() + sample.square()
    unconditional = sample.cos() - 0.25 * sample
    conditional_vjp = torch.autograd.grad(
        conditional, sample, grad_outputs=error, retain_graph=True
    )[0]
    unconditional_vjp = torch.autograd.grad(unconditional, sample, grad_outputs=error)[
        0
    ]
    split = endpoint_state_guidance_from_cfg_vjps(
        velocity=velocity,
        error=error,
        mask=mask,
        conditional_vjp=conditional_vjp,
        unconditional_vjp=unconditional_vjp,
        sigma=sigma,
        guide_scale=guide_scale,
        beta=3.0,
        kp=0.05,
    )

    torch.testing.assert_close(split.velocity, combined.velocity)


def test_feedback_groups_use_four_distinct_measured_frames():
    frames = [
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8))
        for value in (10, 20, 30, 40, 50, 60, 70, 80)
    ]
    events = build_feedback_events(
        future_frames=frames,
        release_steps=[2, 4],
        width=8,
        height=8,
    )

    assert [event.slot for event in events] == [1, 2]
    assert [event.release_step for event in events] == [2, 4]
    assert all(event.frames.shape == (3, 4, 8, 8) for event in events)
    first_means = events[0].frames.mean(dim=(0, 2, 3))
    assert torch.all(first_means[1:] > first_means[:-1])


def test_even_release_schedule_is_deterministic_and_internal():
    assert evenly_spaced_release_steps(2, 12) == [4, 8]
    with pytest.raises(ValueError, match="fewer than solver steps"):
        evenly_spaced_release_steps(2, 2)


def test_sdpa_fallback_preserves_dtype_and_supports_vjp(monkeypatch):
    monkeypatch.setattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False)
    query = torch.randn(1, 3, 2, 4, requires_grad=True)
    key = torch.randn(1, 3, 2, 4, requires_grad=True)
    value = torch.randn(1, 3, 2, 4, requires_grad=True)

    output = attention_module.attention(
        query,
        key,
        value,
        k_lens=torch.tensor([2]),
        dtype=torch.float32,
    )
    output.sum().backward()

    assert output.shape == query.shape
    assert output.dtype == query.dtype
    assert query.grad is not None


def test_wan_block_checkpointing_preserves_input_vjp(monkeypatch):
    monkeypatch.setattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False)
    monkeypatch.setattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False)
    model = WanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=2,
        in_dim=2,
        dim=12,
        ffn_dim=24,
        freq_dim=8,
        text_dim=8,
        out_dim=2,
        num_heads=2,
        num_layers=2,
    ).eval().requires_grad_(False)
    context = [torch.randn(2, 8)]
    timestep = torch.tensor([500.0])

    def input_vjp(checkpointing: bool) -> tuple[torch.Tensor, torch.Tensor]:
        sample = torch.randn(2, 1, 4, 4, requires_grad=True)
        model.gradient_checkpointing = checkpointing
        output = model([sample], timestep, context, seq_len=4)[0]
        gradient = torch.autograd.grad(output.square().sum(), sample)[0]
        return output.detach(), gradient

    torch.manual_seed(0)
    native_output, native_vjp = input_vjp(False)
    torch.manual_seed(0)
    checkpoint_output, checkpoint_vjp = input_vjp(True)

    torch.testing.assert_close(checkpoint_output, native_output)
    torch.testing.assert_close(checkpoint_vjp, native_vjp)
