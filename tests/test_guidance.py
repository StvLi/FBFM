import torch

from dreamzero_fbfm.guidance import guidance_weight, joint_fbfm_guidance


def test_joint_vjp_matches_coupled_linear_endpoint():
    sigma = 0.6
    beta = 10.0
    video = torch.tensor([[0.4, -0.2]], dtype=torch.float64, requires_grad=True)
    action = torch.tensor([[0.1, 0.7]], dtype=torch.float64, requires_grad=True)
    video_velocity = 0.2 * video + 0.3 * action
    action_velocity = 0.4 * video - 0.1 * action
    video_target = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    action_target = torch.tensor([[0.5, -0.3]], dtype=torch.float64)
    video_mask = torch.tensor([[1.0, 0.25]], dtype=torch.float64)
    action_mask = torch.tensor([[0.5, 1.0]], dtype=torch.float64)

    result = joint_fbfm_guidance(
        video_sample=video,
        action_sample=action,
        video_velocity=video_velocity,
        action_velocity=action_velocity,
        video_target=video_target,
        video_mask=video_mask,
        action_target=action_target,
        action_mask=action_mask,
        sigma=sigma,
        beta=beta,
    )

    endpoint_video = video.detach() - sigma * video_velocity.detach()
    endpoint_action = action.detach() - sigma * action_velocity.detach()
    state_error = video_mask * (video_target - endpoint_video)
    action_error = action_mask * (action_target - endpoint_action)
    video_correction = (1 - 0.2 * sigma) * state_error + (-0.4 * sigma) * action_error
    action_correction = (-0.3 * sigma) * state_error + (1 + 0.1 * sigma) * action_error
    weight = guidance_weight(sigma, beta).to(torch.float64)
    torch.testing.assert_close(
        result.video_velocity, video_velocity.detach() - weight * video_correction
    )
    torch.testing.assert_close(
        result.action_velocity, action_velocity.detach() - weight * action_correction
    )
    assert result.diagnostics["guided"] is True
    assert not result.video_velocity.requires_grad
    assert not result.action_velocity.requires_grad


def test_zero_masks_are_exact_baseline_and_detached():
    video = torch.randn(1, 2, requires_grad=True)
    action = torch.randn(1, 2, requires_grad=True)
    video_velocity = video * 0.2
    action_velocity = action * 0.3
    result = joint_fbfm_guidance(
        video_sample=video,
        action_sample=action,
        video_velocity=video_velocity,
        action_velocity=action_velocity,
        video_target=torch.zeros_like(video),
        video_mask=torch.zeros_like(video),
        action_target=torch.zeros_like(action),
        action_mask=torch.zeros_like(action),
        sigma=0.5,
    )
    torch.testing.assert_close(result.video_velocity, video_velocity.detach())
    torch.testing.assert_close(result.action_velocity, action_velocity.detach())
    assert result.diagnostics["guided"] is False
