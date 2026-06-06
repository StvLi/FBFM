from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]
WAN_VA_ROOT = Path(__file__).resolve().parents[1] / "wan_va"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from fbfm.policies.fbfm.configuration_rtc import RTCConfig
from lingbot_va_bridge import FeedbackStateBuffer
from lingbot_va_bridge import FeedbackSlotTracker
from lingbot_va_bridge import SlotAlignedStateBuffer
from lingbot_va_bridge import VA_PrevChunkAdapter
from lingbot_va_bridge import WrapperedFlowMatchScheduler
from utils import FlowMatchScheduler
from wan_va_server import VA_Server


def test_wrapper_scheduler_matches_base_without_guidance():
    rtc_config = RTCConfig()
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        extra_one_step=True,
        rtc_config=rtc_config,
    )
    base = FlowMatchScheduler(
        num_inference_steps=4,
        extra_one_step=True,
    )
    base.set_timesteps(4)
    wrapped.set_timesteps(4)

    x_t = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    sample = x_t.clone()
    timestep = wrapped.timesteps[0]

    def denoise_fn(x):
        return 2 * x + 1

    wrapped_out = wrapped.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=sample,
        constrained_y=None,
        weights=None,
    )
    base_out = base.step(denoise_fn(x_t), timestep=timestep, sample=sample)
    assert torch.allclose(wrapped_out, base_out)


def test_wrapper_scheduler_guidance_changes_with_target_and_weights():
    rtc_config = RTCConfig(max_guidance_weight=10.0)
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        rtc_config=rtc_config,
    )
    wrapped.set_timesteps(4)

    x_t = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    sample = x_t.clone()
    timestep = wrapped.timesteps[0]

    def denoise_fn(x):
        return x.square() + 0.5 * x

    out_a = wrapped.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=sample,
        constrained_y=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )
    out_b = wrapped.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=sample,
        constrained_y=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        weights=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
    )
    out_c = wrapped.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=sample,
        constrained_y=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        weights=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    assert not torch.allclose(out_a, out_b)
    assert not torch.allclose(out_a, out_c)


def test_feedback_state_buffer_exports_recent_states():
    buffer = FeedbackStateBuffer(
        state_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    buffer.append_vectors(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32))
    buffer.append_vectors(torch.tensor([9, 10, 11, 12], dtype=torch.float32))

    recent, count = buffer.export_recent(2)
    assert count == 2
    assert recent.shape == (2, 4)
    assert torch.equal(recent[0], torch.tensor([5, 6, 7, 8], dtype=torch.float32))
    assert torch.equal(recent[1], torch.tensor([9, 10, 11, 12], dtype=torch.float32))


def test_feedback_slot_tracker_emits_one_slot_every_four_observations():
    tracker = FeedbackSlotTracker(obs_per_state=4)
    assert tracker.append(1) == 0
    assert tracker.append(2) == 0
    assert tracker.append(1) == 1
    assert tracker.obs_count == 4
    assert tracker.append(4) == 1
    assert tracker.obs_count == 8
    tracker.reset()
    assert tracker.obs_count == 0


def test_slot_aligned_state_buffer_exports_state_and_mask():
    buffer = SlotAlignedStateBuffer(
        state_dim=3,
        slot_count=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert buffer.append_state(torch.tensor([1.0, 2.0, 3.0]))
    assert buffer.append_state(torch.tensor([4.0, 5.0, 6.0]))
    assert not buffer.append_state(torch.tensor([7.0, 8.0, 9.0]))

    states, mask, count = buffer.export()
    assert count == 2
    assert states.shape == (2, 3)
    assert torch.equal(mask, torch.tensor([1.0, 1.0]))
    assert torch.equal(states[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(states[1], torch.tensor([4.0, 5.0, 6.0]))


def test_prev_chunk_adapter_uses_inference_delay_tail_as_prefix():
    prev_actions = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)
    adapter = VA_PrevChunkAdapter(
        constrain_mode="Feedback",
        prev_actions=prev_actions,
        used_action_channel_ids=[0, 1],
        action_num=8,
        action_dim=2,
        frame_chunk_size=2,
        action_per_frame=4,
        state_num=2,
        latent_channel=1,
        latent_height=1,
        latent_width=1,
        state_dim=1,
        prev_states=None,
        prev_state_constrained_num=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_delay=3,
    )

    assert adapter.action_constrained_num == 3
    constrained = adapter.get_constrained_actions()[0, :, :, :, 0].permute(1, 2, 0).reshape(-1, 2)
    expected_tail = torch.tensor(
        [
            [5.0, 13.0],
            [6.0, 14.0],
            [7.0, 15.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(constrained[:3], expected_tail)


def test_prev_chunk_adapter_uses_explicit_state_mask_not_prefix_mask():
    prev_states = torch.tensor(
        [
            [1.0],
            [2.0],
        ],
        dtype=torch.float32,
    )
    prev_state_mask = torch.tensor([0.0, 1.0], dtype=torch.float32)
    adapter = VA_PrevChunkAdapter(
        constrain_mode="Feedback",
        prev_actions=None,
        used_action_channel_ids=[0],
        action_num=2,
        action_dim=1,
        frame_chunk_size=2,
        action_per_frame=1,
        state_num=2,
        latent_channel=1,
        latent_height=1,
        latent_width=1,
        state_dim=1,
        prev_states=prev_states,
        prev_state_constrained_num=2,
        prev_state_mask=prev_state_mask,
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_delay=0,
    )

    weights = adapter.get_state_prefix_weights()[0, 0, :, 0, 0]
    constrained_states = adapter.get_constrained_states()[0, 0, :, 0, 0]
    assert torch.equal(weights, prev_state_mask)
    assert torch.equal(constrained_states, torch.tensor([1.0, 2.0], dtype=torch.float32))


def _make_fake_feedback_server(latent_frames=1):
    server = object.__new__(VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.frame_st_id = 0
    server.latent_height = 1
    server.latent_width = 1
    server.job_config = SimpleNamespace(frame_chunk_size=2, feedback_obs_per_state=4)
    server.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=1))
    server.feedback_streaming_vae = object()
    server.feedback_streaming_vae_half = None
    server.feedback_obs_per_state = 4
    server.feedback_state_buffer = SlotAlignedStateBuffer(
        state_dim=1,
        slot_count=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    server.feedback_slot_tracker = FeedbackSlotTracker(obs_per_state=4)
    server.feedback_pending_obs = []
    server.feedback_stream_seeded = False
    server.feedback_has_received_window = False
    server.feedback_constraint_active = False
    server.feedback_target_frame_st_id = 0

    encode_calls = []

    def fake_encode(obs, streaming_vae, streaming_vae_half):
        del streaming_vae, streaming_vae_half
        obs_chunk = list(obs["obs"])
        encode_calls.append(obs_chunk)
        frame_count = latent_frames(len(obs_chunk)) if callable(latent_frames) else latent_frames
        return torch.full((1, 1, frame_count, 1, 1), float(len(encode_calls)))

    server._encode_obs_with_stream_wrappers = fake_encode
    return server, encode_calls


def test_feedback_stream_aligns_four_new_observations_to_one_state_slot():
    server, encode_calls = _make_fake_feedback_server()

    server._prime_feedback_stream_initial_obs({"obs": "init"})
    assert encode_calls == [["init"]]

    server._feedback({"obs": ["init", "obs4", "obs8", "obs12"]})
    assert encode_calls == [["init"]]
    assert server.feedback_pending_obs == ["obs4", "obs8", "obs12"]
    assert len(server.feedback_state_buffer) == 0

    server._feedback({"obs": ["obs4", "obs8", "obs12", "obs16"]})
    assert encode_calls[-1] == ["obs4", "obs8", "obs12", "obs16"]
    assert len(server.feedback_state_buffer) == 0
    assert server.feedback_pending_obs == []

    server.frame_st_id = 2
    server._start_feedback_accumulation_window()
    assert server.feedback_target_frame_st_id == 2

    server._feedback({"obs": ["obs8", "obs12", "obs16", "obs20"]})
    server._feedback({"obs": ["obs12", "obs16", "obs20", "obs24"]})
    server._feedback({"obs": ["obs16", "obs20", "obs24", "obs28"]})
    assert len(encode_calls) == 2
    assert len(server.feedback_state_buffer) == 0

    server._feedback({"obs": ["obs20", "obs24", "obs28", "obs32"]})
    assert encode_calls[-1] == ["obs20", "obs24", "obs28", "obs32"]

    states, mask, count = server.feedback_state_buffer.export()
    assert count == 1
    assert torch.equal(mask, torch.tensor([1.0, 0.0]))
    assert torch.equal(states[:, 0], torch.tensor([3.0, 0.0]))

    server._feedback({"obs": ["obs24", "obs28", "obs32", "obs36"]})
    server._feedback({"obs": ["obs28", "obs32", "obs36", "obs40"]})
    server._feedback({"obs": ["obs32", "obs36", "obs40", "obs44"]})
    server._feedback({"obs": ["obs36", "obs40", "obs44", "obs48"]})
    states, mask, count = server.feedback_state_buffer.export()
    assert count == 2
    assert torch.equal(mask, torch.tensor([1.0, 1.0]))
    assert torch.equal(states[:, 0], torch.tensor([3.0, 4.0]))

    server._feedback({"obs": ["obs40", "obs44", "obs48", "obs52"]})
    assert encode_calls[-1] == ["obs36", "obs40", "obs44", "obs48"]
    assert server.feedback_pending_obs == []


def test_feedback_stream_rejects_latent_frame_count_mismatch():
    server, _ = _make_fake_feedback_server(
        latent_frames=lambda obs_count: 2 if obs_count == 4 else 1
    )
    server.feedback_stream_seeded = True
    server.feedback_has_received_window = True
    server.feedback_constraint_active = True
    server.feedback_pending_obs = ["obs20", "obs24", "obs28"]

    try:
        server._feedback({"obs": ["obs20", "obs24", "obs28", "obs32"]})
    except RuntimeError as exc:
        assert "feedback latent/frame count mismatch" in str(exc)
    else:
        raise AssertionError("expected feedback latent/frame count mismatch")
