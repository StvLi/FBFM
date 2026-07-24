from types import SimpleNamespace

import numpy as np
import torch

from dreamzero_fbfm.constraints import ActionNormalizer
from dreamzero_fbfm.runtime import (
    DreamZeroFBFMRuntime,
    DreamZeroFeedbackEncoder,
    FeedbackObservation,
)


class FakeHead:
    def __init__(self):
        self.action_horizon = 16
        self.num_inference_steps = 16
        self.cfg_scale = 5.0
        self.model = SimpleNamespace(action_dim=32)
        self.scheduler = SimpleNamespace(num_train_timesteps=1000)
        self.dit_step_mask = [
            True, True, True, False, False, False, True, False,
            False, False, True, False, False, True, True, True,
        ]

        def denoise(**kwargs):
            video = kwargs["noisy_input"]
            action = kwargs["action"]
            action_mean = action.mean().expand_as(video)
            video_mean = video.mean().expand_as(action)
            conditional_video = 0.2 * video + 0.1 * action_mean
            conditional_action = 0.3 * action + 0.1 * video_mean
            unconditional_video = 0.1 * video
            unconditional_action = torch.zeros_like(action)
            return [
                (conditional_video, conditional_action),
                (unconditional_video, unconditional_action),
            ]

        self._run_diffusion_steps = denoise


class FakePolicy:
    def __init__(self):
        self.action_head = FakeHead()


def _feedback(offset: int) -> FeedbackObservation:
    return FeedbackObservation(
        action_offset=offset,
        main_image=np.zeros((1, 1, 3), dtype=np.uint8),
        wrist_image=np.zeros((1, 1, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        task_description="test",
    )


def test_feedback_encoder_prepends_causal_anchor():
    encoder = object.__new__(DreamZeroFeedbackEncoder)
    encoder.stride = 2
    encoder.observations_per_latent = 4
    encoder.reset()
    encoder._transform_frame = lambda item: torch.full((1, 1, 1), item.action_offset)
    windows = []

    def encode(images):
        windows.append(images.clone())
        return images[:, -1]

    encoder._encode = encode
    encoder.set_anchor(_feedback(0))
    outputs = []
    for offset in range(1, 9):
        outputs.extend(encoder.add(_feedback(offset)))

    assert len(outputs) == 1
    torch.testing.assert_close(windows[0].reshape(-1), torch.tensor([0, 2, 4, 6, 8]))


def test_runtime_hook_guides_action_and_detaches_solver_graph():
    policy = FakePolicy()
    normalizer = ActionNormalizer(
        torch.full((7,), -1.0), torch.full((7,), 1.0), model_dim=32
    )
    runtime = DreamZeroFBFMRuntime(policy, normalizer, mode="RTC")
    runtime.begin_chunk(np.full((8, 7), 0.5, dtype=np.float32), pseudo_async=False)
    video = torch.randn(1, 2, 2, 1, 1)
    action = torch.randn(1, 16, 32)
    baseline = policy.action_head._run_diffusion_steps(
        noisy_input=video,
        action=action,
        kv_cache_metadata={"update_kv_cache": True},
    )
    guided = policy.action_head._run_diffusion_steps(
        noisy_input=video,
        action=action,
        timestep=torch.full((1, 2), 600),
        timestep_action=torch.full((1, 16), 600),
        kv_cache_metadata={"update_kv_cache": False},
    )
    assert not torch.equal(guided[0][1], baseline[0][1])
    assert not guided[0][0].requires_grad
    assert not guided[0][1].requires_grad
    assert policy.action_head.dit_step_mask == [
        True, True, True, False, False, False, True, False,
        False, False, True, False, False, True, True, True,
    ]


def test_none_preserves_native_solver_output_exactly():
    policy = FakePolicy()
    normalizer = ActionNormalizer(
        torch.full((7,), -1.0), torch.full((7,), 1.0), model_dim=32
    )
    original = policy.action_head._run_diffusion_steps
    runtime = DreamZeroFBFMRuntime(policy, normalizer, mode="NONE")
    runtime.begin_chunk(np.full((8, 7), 0.5, dtype=np.float32), pseudo_async=False)
    video = torch.randn(1, 2, 2, 1, 1)
    action = torch.randn(1, 16, 32)
    kwargs = {
        "noisy_input": video,
        "action": action,
        "timestep": torch.full((1, 2), 600),
        "timestep_action": torch.full((1, 16), 600),
        "kv_cache_metadata": {"update_kv_cache": False},
    }
    expected = original(**kwargs)
    actual = policy.action_head._run_diffusion_steps(**kwargs)
    for actual_branch, expected_branch in zip(actual, expected):
        torch.testing.assert_close(actual_branch[0], expected_branch[0], rtol=0, atol=0)
        torch.testing.assert_close(actual_branch[1], expected_branch[1], rtol=0, atol=0)
