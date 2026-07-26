import json
from types import SimpleNamespace

import numpy as np
import torch

import dreamzero_fbfm.runtime as runtime_module
from dreamzero_fbfm.constraints import ActionNormalizer
from dreamzero_fbfm.runtime import (
    DreamZeroFBFMRuntime,
    DreamZeroFeedbackEncoder,
    FeedbackObservation,
)
from dreamzero_fbfm.settings import DEFAULT_STATE_WEIGHT


class FakeHead:
    def __init__(self):
        self.action_horizon = 16
        self.num_frame_per_block = 2
        self.num_inference_steps = 16
        self.cfg_scale = 5.0
        self.supports_external_step_guidance = True
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


def test_feedback_encoder_uses_training_aligned_stride_three_history():
    encoder = object.__new__(DreamZeroFeedbackEncoder)
    encoder.observations_per_latent = 4
    encoder.actions_per_latent = 12
    encoder.observation_interval = 3
    encoder.latent_slots = 2
    encoder.reset()
    encoder._transform_frame = lambda item: torch.full((1, 1, 1), item.action_offset)
    windows = []

    def encode(images):
        windows.append(images.clone())
        return images[:, -1]

    encoder._encode = encode
    encoder.set_anchor(_feedback(0))
    outputs = []
    for offset in range(1, 13):
        outputs.extend(encoder.add(_feedback(offset)))

    assert [output.action_offset for output in outputs] == [3, 6, 9, 12]
    assert [output.slot for output in outputs] == [0] * 4
    assert [output.complete for output in outputs] == [False] * 3 + [True]
    expected_windows = [
        [0, 0, 0, 0, 3],
        [0, 0, 0, 3, 6],
        [0, 0, 3, 6, 9],
        [0, 3, 6, 9, 12],
    ]
    assert [window.reshape(-1).tolist() for window in windows] == expected_windows
    assert outputs[-1].source_offsets == (0, 3, 6, 9, 12)


def test_feedback_encoder_refreshes_second_latent_slot():
    encoder = object.__new__(DreamZeroFeedbackEncoder)
    encoder.observations_per_latent = 4
    encoder.actions_per_latent = 12
    encoder.observation_interval = 3
    encoder.latent_slots = 2
    encoder.reset()
    encoder._transform_frame = lambda item: torch.full((1, 1, 1), item.action_offset)
    encoder._encode = lambda images: images[:, -1]
    encoder.set_anchor(_feedback(0))

    outputs = []
    for offset in range(1, 25):
        outputs.extend(encoder.add(_feedback(offset)))

    assert [output.slot for output in outputs] == [0] * 4 + [1] * 4
    assert outputs[4].source_offsets == (12, 12, 12, 12, 15)
    assert outputs[-1].source_offsets == (12, 15, 18, 21, 24)


def test_runtime_updates_state_only_on_training_aligned_feedback(tmp_path):
    policy = FakePolicy()
    normalizer = ActionNormalizer(
        torch.full((7,), -1.0), torch.full((7,), 1.0), model_dim=32
    )
    audit_path = tmp_path / "audit.jsonl"
    runtime = DreamZeroFBFMRuntime(
        policy, normalizer, mode="FBFM", audit_path=str(audit_path)
    )
    runtime.begin_chunk(
        np.zeros((8, 7), dtype=np.float32), pseudo_async=False
    )

    encoder = runtime.feedback_encoder
    encoder._transform_frame = lambda item: torch.full(
        (1, 1, 2, 1, 1), float(item.action_offset)
    )
    encoder._encode = lambda images: images[:, -1]
    encoder.set_anchor(_feedback(0))

    video = torch.randn(1, 2, 2, 1, 1)
    action = torch.randn(1, 16, 32)
    for offset in range(1, 9):
        runtime.submit_feedback(_feedback(offset))
        policy.action_head._run_diffusion_steps(
            noisy_input=video,
            action=action,
            timestep=torch.full((1, 2), 600),
            timestep_action=torch.full((1, 16), 600),
            kv_cache_metadata={"update_kv_cache": False},
        )
        assert runtime._constraints is not None
        assert runtime._constraints.version == offset // 3

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    solver_records = [record for record in records if record["event"] == "solver_step"]
    assert [record["context_version"] for record in solver_records] == [
        0, 0, 1, 1, 1, 2, 2, 2
    ]
    assert [record["feedback_action_offsets"] for record in solver_records] == [
        [], [], [3], [], [], [6], [], []
    ]
    assert [record["feedback_state_slots"] for record in solver_records] == [
        [], [], [0], [], [], [0], [], []
    ]
    assert [record["state_target_updates"] for record in solver_records] == [
        0, 0, 1, 0, 0, 1, 0, 0
    ]


def test_runtime_default_uses_l1_mass_balanced_state_mask(monkeypatch):
    assert DEFAULT_STATE_WEIGHT == 56 / 9600
    policy = FakePolicy()
    normalizer = ActionNormalizer(
        torch.full((7,), -1.0), torch.full((7,), 1.0), model_dim=32
    )
    runtime = DreamZeroFBFMRuntime(policy, normalizer, mode="FBFM")
    assert runtime.state_weight == DEFAULT_STATE_WEIGHT
    runtime.begin_chunk(np.zeros((8, 7), dtype=np.float32), pseudo_async=False)

    encoder = runtime.feedback_encoder
    encoder._transform_frame = lambda item: torch.full(
        (1, 1, 2, 1, 1), float(item.action_offset)
    )
    encoder._encode = lambda images: images[:, -1]
    encoder.set_anchor(_feedback(0))
    for offset in range(1, 4):
        runtime.submit_feedback(_feedback(offset))

    captured = {}
    original_guidance = runtime_module.joint_fbfm_guidance

    def capture_guidance(**kwargs):
        captured["video_mask"] = kwargs["video_mask"].detach().clone()
        return original_guidance(**kwargs)

    monkeypatch.setattr(runtime_module, "joint_fbfm_guidance", capture_guidance)
    policy.action_head._run_diffusion_steps(
        noisy_input=torch.randn(1, 2, 2, 1, 1),
        action=torch.randn(1, 16, 32),
        timestep=torch.full((1, 2), 600),
        timestep_action=torch.full((1, 16), 600),
        kv_cache_metadata={"update_kv_cache": False},
    )

    expected = torch.tensor([[[[[DEFAULT_STATE_WEIGHT]], [[0.0]]]]])
    torch.testing.assert_close(captured["video_mask"], expected, rtol=0, atol=0)


def test_runtime_guides_current_step_without_polluting_native_cache(tmp_path):
    policy = FakePolicy()
    normalizer = ActionNormalizer(
        torch.full((7,), -1.0), torch.full((7,), 1.0), model_dim=32
    )
    audit_path = tmp_path / "audit.jsonl"
    runtime = DreamZeroFBFMRuntime(
        policy, normalizer, mode="RTC", audit_path=str(audit_path)
    )
    runtime.begin_chunk(np.full((8, 7), 0.5, dtype=np.float32), pseudo_async=False)
    video = torch.randn(1, 2, 2, 1, 1)
    action = torch.randn(1, 16, 32)
    baseline = policy.action_head._run_diffusion_steps(
        noisy_input=video,
        action=action,
        kv_cache_metadata={"update_kv_cache": True},
    )
    cached = policy.action_head._run_diffusion_steps(
        noisy_input=video,
        action=action,
        timestep=torch.full((1, 2), 600),
        timestep_action=torch.full((1, 16), 600),
        kv_cache_metadata={"update_kv_cache": False},
    )
    for cached_branch, baseline_branch in zip(cached, baseline):
        torch.testing.assert_close(cached_branch[0], baseline_branch[0])
        torch.testing.assert_close(cached_branch[1], baseline_branch[1])

    base_video = cached[1][0] + policy.action_head.cfg_scale * (
        cached[0][0] - cached[1][0]
    )
    base_action = cached[0][1]
    guided_video, guided_action = policy.action_head.external_step_guidance(
        scheduler_index=0,
        model_evaluated=True,
        video_sample=video,
        action_sample=action,
        video_velocity=base_video,
        action_velocity=base_action,
        timestep_action=torch.full((1, 16), 600),
    )
    assert not torch.equal(guided_action, base_action)
    assert not guided_video.requires_grad
    assert not guided_action.requires_grad

    skipped_video, skipped_action = policy.action_head.external_step_guidance(
        scheduler_index=3,
        model_evaluated=False,
        video_sample=video + 0.1,
        action_sample=action - 0.2,
        video_velocity=base_video,
        action_velocity=base_action,
        timestep_action=torch.full((1, 16), 400),
    )
    assert not torch.equal(skipped_action, guided_action)
    assert not skipped_video.requires_grad
    assert not skipped_action.requires_grad
    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    scheduler_records = [
        record for record in records if record["event"] == "scheduler_guidance_step"
    ]
    assert [record["model_evaluated"] for record in scheduler_records] == [True, False]
    assert scheduler_records[1]["jacobian_reused"] is True
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
