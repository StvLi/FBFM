import pathlib
import sys

import pytest
import torch


DREAMZERO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DREAMZERO_ROOT))

from groot.vla.model.dreamzero.modules.fbfm_guidance import (  # noqa: E402
    ConstraintMode,
    FBFMGuidanceConfig,
    JointConstraints,
    apply_joint_guidance,
    clean_prediction,
    prefix_constraint,
    prepare_guided_inputs,
)


def _joint_flow(video: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    action_context = action.mean(dim=(1, 2), keepdim=True).view(action.shape[0], 1, 1, 1, 1)
    video_context = video.mean(dim=(1, 2, 3, 4), keepdim=True).view(video.shape[0], 1, 1)
    return 0.5 * video + action_context, 0.25 * action + video_context


def _guided(mode: ConstraintMode, video_target: float, action_target: float):
    video, action = prepare_guided_inputs(
        torch.full((1, 2, 1, 2, 2), 0.4),
        torch.full((1, 3, 2), -0.2),
    )
    video_flow, action_flow = _joint_flow(video, action)
    constraints = JointConstraints(
        video_target=torch.full_like(video, video_target),
        video_mask=torch.ones_like(video),
        action_target=torch.full_like(action, action_target),
        action_mask=torch.ones_like(action),
    )
    return apply_joint_guidance(
        video_sample=video,
        action_sample=action,
        video_flow=video_flow,
        action_flow=action_flow,
        video_local_flow=video_flow,
        action_local_flow=action_flow,
        video_sigma=0.7,
        action_sigma=0.7,
        constraints=constraints,
        config=FBFMGuidanceConfig(mode=mode, max_guidance_weight=5.0),
    ), (video_flow, action_flow)


def test_constraint_mode_parsing():
    assert ConstraintMode.parse("none") is ConstraintMode.NONE
    assert ConstraintMode.parse("RTC") is ConstraintMode.RTC
    assert ConstraintMode.parse("feedback") is ConstraintMode.FEEDBACK
    with pytest.raises(ValueError):
        ConstraintMode.parse("bad-mode")


def test_prefix_constraint_uses_tail_as_next_prefix():
    sample = torch.zeros(1, 5, 2)
    previous_tail = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    target, mask = prefix_constraint(sample, previous_tail, time_dim=1)
    torch.testing.assert_close(target, previous_tail[:, -5:])
    assert torch.equal(mask, torch.ones_like(mask))


def test_none_mode_is_numerically_identical_and_does_not_require_grad():
    video = torch.randn(1, 2, 1, 2, 2)
    action = torch.randn(1, 3, 2)
    video_flow = torch.randn_like(video)
    action_flow = torch.randn_like(action)
    result = apply_joint_guidance(
        video_sample=video,
        action_sample=action,
        video_flow=video_flow,
        action_flow=action_flow,
        video_local_flow=video_flow,
        action_local_flow=action_flow,
        video_sigma=0.5,
        action_sigma=0.5,
        constraints=JointConstraints(),
        config=FBFMGuidanceConfig(mode=ConstraintMode.NONE),
    )
    assert result[0] is video_flow
    assert result[1] is action_flow


def test_feedback_vjp_is_finite_nonzero_and_target_sensitive():
    (guided_a, base_a) = _guided(ConstraintMode.FEEDBACK, video_target=1.0, action_target=0.5)
    (guided_b, _) = _guided(ConstraintMode.FEEDBACK, video_target=-1.0, action_target=-0.5)
    for guided, base in zip(guided_a, base_a):
        assert torch.isfinite(guided).all()
        assert not torch.equal(guided, base)
    assert not torch.equal(guided_a[0], guided_b[0])
    assert not torch.equal(guided_a[1], guided_b[1])


def test_rtc_ignores_video_target_but_keeps_joint_action_vjp():
    (guided_a, base) = _guided(ConstraintMode.RTC, video_target=100.0, action_target=0.5)
    (guided_b, _) = _guided(ConstraintMode.RTC, video_target=-100.0, action_target=0.5)
    torch.testing.assert_close(guided_a[0], guided_b[0])
    torch.testing.assert_close(guided_a[1], guided_b[1])
    # Action clean prediction depends on video input in _joint_flow, so action-only
    # RTC legitimately corrects both components of the joint state.
    assert not torch.equal(guided_a[0], base[0])
    assert not torch.equal(guided_a[1], base[1])


def test_clean_prediction_matches_flow_unipc_convention():
    sample = torch.tensor([2.0])
    flow = torch.tensor([0.5])
    torch.testing.assert_close(clean_prediction(sample, flow, 0.4), torch.tensor([1.8]))


def test_two_rank_cfg_partial_vjps_sum_to_single_rank_joint_vjp():
    cfg = 3.0
    constraints = JointConstraints(
        video_target=torch.full((1, 2, 1, 2, 2), 0.7),
        video_mask=torch.ones((1, 2, 1, 2, 2)),
        action_target=torch.full((1, 3, 2), -0.4),
        action_mask=torch.ones((1, 3, 2)),
    )
    config = FBFMGuidanceConfig(mode=ConstraintMode.FEEDBACK, max_guidance_weight=5.0)

    def branches(video, action):
        action_context = action.mean().reshape(1, 1, 1, 1, 1)
        video_context = video.mean().reshape(1, 1, 1)
        cond_video = 0.4 * video + action_context
        uncond_video = -0.2 * video + 0.5 * action_context
        cond_action = 0.3 * action + video_context
        global_video = uncond_video + cfg * (cond_video - uncond_video)
        return cond_video, uncond_video, cond_action, global_video

    video_full, action_full = prepare_guided_inputs(
        torch.full((1, 2, 1, 2, 2), 0.1),
        torch.full((1, 3, 2), -0.1),
    )
    _, _, action_flow_full, video_flow_full = branches(video_full, action_full)
    guided_full_video, guided_full_action = apply_joint_guidance(
        video_sample=video_full,
        action_sample=action_full,
        video_flow=video_flow_full,
        action_flow=action_flow_full,
        video_local_flow=video_flow_full,
        action_local_flow=action_flow_full,
        video_sigma=0.6,
        action_sigma=0.6,
        constraints=constraints,
        config=config,
    )

    partials = []
    for rank in (0, 1):
        video, action = prepare_guided_inputs(video_full, action_full)
        cond_video, uncond_video, cond_action, global_video = branches(video, action)
        local_video = cfg * cond_video if rank == 0 else (1.0 - cfg) * uncond_video
        local_action = cond_action if rank == 0 else torch.zeros_like(cond_action)
        guided_video, guided_action = apply_joint_guidance(
            video_sample=video,
            action_sample=action,
            video_flow=global_video,
            action_flow=cond_action,
            video_local_flow=local_video,
            action_local_flow=local_action,
            video_sigma=0.6,
            action_sigma=0.6,
            constraints=constraints,
            config=config,
            world_size=2,
            reduce_correction=lambda tensor: None,
        )
        partials.append((global_video.detach() - guided_video, cond_action.detach() - guided_action))

    reconstructed_video = video_flow_full.detach() - partials[0][0] - partials[1][0]
    reconstructed_action = action_flow_full.detach() - partials[0][1] - partials[1][1]
    torch.testing.assert_close(reconstructed_video, guided_full_video)
    torch.testing.assert_close(reconstructed_action, guided_full_action)
