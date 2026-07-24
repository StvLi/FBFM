from types import SimpleNamespace

import numpy as np
import torch

from dreamzero_fbfm.constraints import ActionNormalizer
from dreamzero_fbfm.runtime import DreamZeroFBFMRuntime


class FakeHead:
    def __init__(self):
        self.action_horizon = 16
        self.num_inference_steps = 16
        self.cfg_scale = 5.0
        self.model = SimpleNamespace(action_dim=32)
        self.dit_step_mask = [True] * 16

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
        kv_cache_metadata={"update_kv_cache": False},
    )
    assert not torch.equal(guided[0][1], baseline[0][1])
    assert not guided[0][0].requires_grad
    assert not guided[0][1].requires_grad
