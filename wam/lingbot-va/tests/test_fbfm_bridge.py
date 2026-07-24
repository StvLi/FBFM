from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]
LINGBOT_VA_ROOT = Path(__file__).resolve().parents[1]
WAN_VA_ROOT = Path(__file__).resolve().parents[1] / "wan_va"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LINGBOT_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(LINGBOT_VA_ROOT))
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from fbfm.policies.fbfm.configuration_rtc import RTCConfig
from lingbot_va_bridge import FeedbackStateBuffer
from lingbot_va_bridge import FeedbackSlotTracker
from lingbot_va_bridge import SlotAlignedStateBuffer
from lingbot_va_bridge import SolverStepClock
from lingbot_va_bridge import ChunkConstraintContext
from lingbot_va_bridge import ConstraintMode
from lingbot_va_bridge import VA_PrevChunkAdapter
from lingbot_va_bridge import WrapperedFlowMatchScheduler
from lingbot_va_bridge import build_rtc_action_mask
from fbfm.configs.types import RTCAttentionSchedule
from fbfm.policies.fbfm.modeling_rtc import RTCProcessor
from utils import FlowMatchScheduler


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

    wrapped_zero_mask_out = wrapped.step(
        original_denoise_step_partial=denoise_fn,
        x_t=x_t,
        timestep=timestep,
        sample=sample,
        constrained_y=torch.full_like(x_t, 999.0),
        weights=torch.zeros_like(x_t),
    )
    assert torch.equal(wrapped_zero_mask_out, base_out)
    assert wrapped.last_step_diagnostics["has_guidance"] is False
    assert wrapped.last_step_diagnostics["mask_nonzero"] == 0
    assert wrapped.last_step_diagnostics["correction_norm"] == 0.0


def test_wrapper_scheduler_matches_lingbot_cache_write_trajectory():
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        extra_one_step=True,
        sigma_min=0.0,
        rtc_config=RTCConfig(),
    )
    base = FlowMatchScheduler(
        num_inference_steps=4,
        extra_one_step=True,
        sigma_min=0.0,
    )
    wrapped.set_timesteps(4)
    base.set_timesteps(4)
    timesteps = torch.nn.functional.pad(wrapped.timesteps, (0, 1), value=0)

    wrapped_calls = []
    wrapped_sample = torch.tensor([[0.75, -0.25]], dtype=torch.float32)
    for index, timestep in enumerate(timesteps):
        cache_only = index == len(timesteps) - 1

        def wrapped_denoise(x, cache_only=cache_only):
            wrapped_calls.append((cache_only, x.detach().clone()))
            return 0.25 * x + 1.0

        wrapped_sample = wrapped.step(
            original_denoise_step_partial=wrapped_denoise,
            x_t=wrapped_sample,
            timestep=timestep,
            sample=wrapped_sample,
            cache_only=cache_only,
            constrained_y=torch.full_like(wrapped_sample, 999.0),
            weights=(
                torch.ones_like(wrapped_sample)
                if cache_only else torch.zeros_like(wrapped_sample)
            ),
        )

    base_calls = []
    base_sample = torch.tensor([[0.75, -0.25]], dtype=torch.float32)
    for index, timestep in enumerate(timesteps):
        cache_only = index == len(timesteps) - 1
        base_calls.append((cache_only, base_sample.detach().clone()))
        model_output = 0.25 * base_sample + 1.0
        if not cache_only:
            base_sample = base.step(model_output, timestep, base_sample)

    assert len(wrapped_calls) == len(base_calls) == 5
    assert [cache_only for cache_only, _ in wrapped_calls] == [
        cache_only for cache_only, _ in base_calls
    ]
    for (_, wrapped_input), (_, base_input) in zip(wrapped_calls, base_calls):
        assert torch.equal(wrapped_input, base_input)
    assert torch.equal(wrapped_sample, base_sample)
    assert not wrapped_sample.requires_grad
    assert wrapped_sample.grad_fn is None


def test_none_infer_matches_upstream_solver_and_cache_sequence(monkeypatch, tmp_path):
    import wan_va_server as server_module

    class FakeTransformer:
        def __init__(self):
            self.calls = []

        def __call__(self, model_input, update_cache, cache_name, action_mode):
            del cache_name
            x_t = model_input["noisy_latents"]
            self.calls.append((action_mode, update_cache, x_t.detach().clone()))
            velocity = 0.25 * x_t + 1.0
            if action_mode:
                return velocity.squeeze(-1).permute(0, 2, 3, 1).reshape(
                    velocity.shape[0], -1, velocity.shape[1]
                )
            return velocity

    class NoConstraintContext:
        chunk_id = 0

        def snapshot_state_constraints(self):
            return (
                torch.zeros(1, 48, 2, 1, 1),
                torch.zeros(1, 1, 2, 1, 1),
                0,
            )

        def snapshot_action_constraints(self):
            return (
                torch.zeros(1, 2, 2, 2, 1),
                torch.zeros(1, 1, 2, 2, 1),
                0,
            )

    server = object.__new__(server_module.VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.job_config = SimpleNamespace(
        frame_chunk_size=2,
        action_dim=2,
        num_inference_steps=3,
        action_num_inference_steps=2,
        video_exec_step=-1,
        guidance_scale=1.0,
        action_guidance_scale=1.0,
        patch_size=(1, 1, 1),
        constraint_mode="NONE",
    )
    server.latent_height = 1
    server.latent_width = 1
    server.action_per_frame = 2
    server.use_cfg = False
    server.scheduler = WrapperedFlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        rtc_config=RTCConfig(),
    )
    server.action_scheduler = WrapperedFlowMatchScheduler(
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        rtc_config=RTCConfig(),
    )
    server.transformer = FakeTransformer()
    server.active_constraint_context = NoConstraintContext()
    server.action_mask = torch.ones(2, dtype=torch.bool)
    server.cache_name = "pos"
    server.exp_save_root = str(tmp_path)
    server._solver_phase = "video"
    server._cancel_requested = threading.Event()
    server._solver_step_clock = SolverStepClock()
    server._solver_step_clock.start(clock_id=None, enabled=False)
    init_latent = torch.full((1, 48, 1, 1, 1), 0.125)
    server._encode_obs = lambda obs: init_latent.clone()
    server._prime_feedback_stream_initial_obs = lambda obs: (_ for _ in ()).throw(
        AssertionError("NONE must not initialize the feedback VAE stream")
    )
    server._prepare_latent_input = lambda latents=None, actions=None, *args, **kwargs: {
        "latent_res_lst": {"noisy_latents": latents},
        "action_res_lst": {"noisy_latents": actions},
    }
    server._repeat_input_for_cfg = lambda model_input: model_input
    server._drain_live_feedback = lambda: (0, False)
    server._sync_cancel_requested = lambda: False
    server.postprocess_action = lambda actions: actions

    monkeypatch.setattr(
        server_module,
        "data_seq_to_patch",
        lambda patch_size, output, *args, **kwargs: output,
    )
    monkeypatch.setattr(server_module, "save_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(server_module, "tqdm", lambda values: values)
    monkeypatch.setattr(server_module.torch.cuda, "empty_cache", lambda: None)

    torch.manual_seed(1234)
    initial_latents = torch.randn(1, 48, 2, 1, 1)
    initial_actions = torch.randn(1, 2, 2, 2, 1)

    reference_video_scheduler = FlowMatchScheduler(
        shift=1.0, sigma_min=0.0, extra_one_step=True
    )
    reference_action_scheduler = FlowMatchScheduler(
        shift=1.0, sigma_min=0.0, extra_one_step=True
    )
    reference_video_scheduler.set_timesteps(3)
    reference_action_scheduler.set_timesteps(2)
    reference_latents = initial_latents.clone()
    for timestep in reference_video_scheduler.timesteps:
        velocity = 0.25 * reference_latents + 1.0
        reference_latents = reference_video_scheduler.step(
            velocity, timestep, reference_latents
        )
        reference_latents[:, :, :1] = init_latent
    reference_actions = initial_actions.clone()
    for timestep in reference_action_scheduler.timesteps:
        velocity = 0.25 * reference_actions + 1.0
        reference_actions = reference_action_scheduler.step(
            velocity, timestep, reference_actions
        )
        reference_actions[:, :, :1] = 0

    torch.manual_seed(1234)
    actions, latents = server._infer({"obs": []}, frame_st_id=0)

    assert torch.equal(latents, reference_latents)
    assert torch.equal(actions, reference_actions)
    video_calls = [call for call in server.transformer.calls if not call[0]]
    action_calls = [call for call in server.transformer.calls if call[0]]
    assert [update_cache for _, update_cache, _ in video_calls] == [0, 0, 0, 1]
    assert [update_cache for _, update_cache, _ in action_calls] == [0, 0, 1]
    assert torch.equal(video_calls[-1][2], reference_latents)
    assert torch.equal(action_calls[-1][2], reference_actions)


def test_solver_step_clock_waits_for_exact_consumption():
    clock = SolverStepClock()
    clock.start(clock_id="chunk-3", enabled=True)
    result = {}

    def grant():
        result.update(
            clock.grant_and_wait(clock_id="chunk-3", count=2, timeout=1.0)
        )

    worker = threading.Thread(target=grant)
    worker.start()
    for _ in range(1000):
        if clock.snapshot()["solver_steps_issued"] == 2:
            break
        threading.Event().wait(0.001)
    assert clock.snapshot()["solver_steps_issued"] == 2
    assert worker.is_alive()
    assert clock.consume()
    assert worker.is_alive()
    assert clock.consume()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result["solver_clock_accepted"]
    assert result["solver_steps_consumed"] == 2
    assert result["solver_steps_available"] == 0


def test_solver_step_clock_rejects_stale_chunk_grant():
    clock = SolverStepClock()
    clock.start(clock_id="current", enabled=True)
    result = clock.grant_and_wait(clock_id="stale", count=1, timeout=0.01)
    assert not result["solver_clock_accepted"]
    assert result["solver_clock_reason"] == "clock_id_mismatch"
    assert result["solver_steps_issued"] == 0


def test_wrapper_scheduler_never_returns_an_autograd_graph():
    rtc_config = RTCConfig(max_guidance_weight=10.0)
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        extra_one_step=True,
        rtc_config=rtc_config,
    )
    wrapped.set_timesteps(4)
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def denoise_fn(x):
        return parameter * x.square()

    sample = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    timestep = wrapped.timesteps[0]
    cases = (
        (None, None),
        (torch.ones_like(sample), torch.zeros_like(sample)),
        (torch.ones_like(sample), torch.ones_like(sample)),
    )
    for constrained_y, weights in cases:
        output = wrapped.step(
            original_denoise_step_partial=denoise_fn,
            x_t=sample,
            timestep=timestep,
            sample=sample,
            constrained_y=constrained_y,
            weights=weights,
        )
        assert not output.requires_grad
        assert output.grad_fn is None
        assert parameter.grad is None


def test_repeated_guided_steps_do_not_chain_solver_graphs():
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        extra_one_step=True,
        rtc_config=RTCConfig(max_guidance_weight=10.0),
    )
    wrapped.set_timesteps(4)
    sample = torch.tensor([[0.5, -0.25]], dtype=torch.float32)

    def denoise_fn(x):
        return x.square() + 0.5 * x

    for timestep in wrapped.timesteps:
        sample = wrapped.step(
            original_denoise_step_partial=denoise_fn,
            x_t=sample,
            timestep=timestep,
            sample=sample,
            constrained_y=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            weights=torch.ones_like(sample),
        )
        assert not sample.requires_grad
        assert sample.grad_fn is None


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


def test_sigma_parameterized_guidance_moves_sample_toward_target():
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        rtc_config=RTCConfig(max_guidance_weight=1.0),
    )
    wrapped.set_timesteps(4)
    sample = torch.tensor([[0.0]], dtype=torch.float32)
    target = torch.tensor([[1.0]], dtype=torch.float32)

    output = wrapped.step(
        original_denoise_step_partial=torch.zeros_like,
        x_t=sample,
        timestep=wrapped.timesteps[0],
        sample=sample,
        constrained_y=target,
        weights=torch.ones_like(sample),
    )

    assert 0 < output.item() < target.item()
    diagnostics = wrapped.last_step_diagnostics
    assert diagnostics["has_guidance"] is True
    assert diagnostics["mask_nonzero"] == 1
    assert diagnostics["error_norm"] > 0
    assert diagnostics["correction_norm"] > 0
    assert diagnostics["guided_velocity_norm"] > diagnostics["base_velocity_norm"]


def test_sigma_parameterized_guidance_matches_paper_vjp_equation():
    max_weight = 10.0
    wrapped = WrapperedFlowMatchScheduler(
        num_inference_steps=4,
        shift=1.0,
        sigma_min=0.0,
        extra_one_step=True,
        rtc_config=RTCConfig(max_guidance_weight=max_weight),
    )
    wrapped.set_timesteps(4)

    sample = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    target = torch.tensor([[1.2, 0.3]], dtype=torch.float64)
    mask = torch.tensor([[1.0, 0.25]], dtype=torch.float64)
    timestep = wrapped.timesteps[1]
    sigma = wrapped.sigmas[1].to(dtype=sample.dtype)
    sigma_next = wrapped.sigmas[2].to(dtype=sample.dtype)
    tau = 1 - sigma

    def denoise_fn(x):
        return 0.3 * x + 0.1

    base_velocity = denoise_fn(sample)
    clean_endpoint = sample - sigma * base_velocity
    error = mask * (target - clean_endpoint)
    endpoint_jacobian = 1 - 0.3 * sigma
    correction = endpoint_jacobian * error
    r_squared = sigma.square() / (tau.square() + sigma.square())
    guidance_weight = torch.clamp(
        sigma / (tau * r_squared), max=max_weight
    )
    guided_velocity = base_velocity - guidance_weight * correction
    expected = sample + (sigma_next - sigma) * guided_velocity

    actual = wrapped.step(
        original_denoise_step_partial=denoise_fn,
        x_t=sample,
        timestep=timestep,
        sample=sample,
        constrained_y=target,
        weights=mask,
    )

    assert torch.allclose(actual, expected)
    diagnostics = wrapped.last_step_diagnostics
    torch.testing.assert_close(
        torch.tensor(diagnostics["guidance_weight"], dtype=sample.dtype),
        guidance_weight,
    )
    torch.testing.assert_close(
        torch.tensor(diagnostics["correction_norm"], dtype=sample.dtype),
        correction.norm(),
    )


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


def _make_mode_adapter(mode):
    return VA_PrevChunkAdapter(
        constrain_mode=mode,
        prev_actions=torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        used_action_channel_ids=[0],
        action_num=8,
        action_dim=1,
        frame_chunk_size=2,
        action_per_frame=4,
        state_num=2,
        latent_channel=1,
        latent_height=1,
        latent_width=1,
        state_dim=1,
        prev_states=torch.tensor([[1.0], [2.0]]),
        prev_state_constrained_num=2,
        prev_state_mask=torch.tensor([1.0, 1.0]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_delay=2,
        execution_horizon=3,
        rtc_attention_schedule="EXP",
    )


def test_constraint_modes_only_gate_masks():
    none = _make_mode_adapter("NONE")
    rtc = _make_mode_adapter("RTC")
    fbfm = _make_mode_adapter("FBFM")

    none_state = none.get_state_prefix_weights()
    rtc_state = rtc.get_state_prefix_weights()
    fbfm_state = fbfm.get_state_prefix_weights()
    none_action = none.get_action_prefix_weights()
    rtc_action = rtc.get_action_prefix_weights()
    fbfm_action = fbfm.get_action_prefix_weights()

    assert torch.count_nonzero(none_state) == 0
    assert torch.count_nonzero(none_action) == 0
    assert torch.count_nonzero(rtc_state) == 0
    assert torch.equal(rtc_action, fbfm_action)
    assert torch.equal(fbfm_state[0, 0, :, 0, 0], torch.ones(2))
    assert torch.equal(none.get_constrained_actions(), rtc.get_constrained_actions())
    assert torch.equal(rtc.get_constrained_actions(), fbfm.get_constrained_actions())
    assert torch.equal(none.get_constrained_states(), fbfm.get_constrained_states())


def test_action_constraint_mask_excludes_unused_lingbot_channels():
    import wan_va_server as server_module

    server = object.__new__(server_module.VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.frame_st_id = 4
    server.feedback_target_frame_st_id = 4
    server.action_mask = torch.tensor([True, False, True])
    server.active_constraint_context = None
    adapter = SimpleNamespace(
        constraint_mode=ConstraintMode.RTC,
        _explicit_state_mask=torch.zeros(1),
        _rtc_action_mask=torch.tensor([1.0, 0.5]),
        _frame_chunk_size=1,
        _action_per_frame=2,
        get_constrained_states=lambda: torch.zeros(1, 1, 1, 1, 1),
        get_constrained_actions=lambda: torch.zeros(1, 3, 1, 2, 1),
    )

    context = server._make_constraint_context(adapter)
    _, mask, _ = context.snapshot_action_constraints()

    assert mask.shape == (1, 3, 1, 2, 1)
    assert torch.equal(mask[0, 0, 0, :, 0], torch.tensor([1.0, 0.5]))
    assert torch.count_nonzero(mask[0, 1]) == 0
    assert torch.equal(mask[0, 2, 0, :, 0], torch.tensor([1.0, 0.5]))


def test_rtc_action_mask_matches_core_implementation():
    for schedule in (RTCAttentionSchedule.LINEAR, RTCAttentionSchedule.EXP):
        expected = RTCProcessor(
            RTCConfig(prefix_attention_schedule=schedule)
        ).get_prefix_weights(start=4, end=20, total=32)
        actual = build_rtc_action_mask(
            total=32,
            inference_delay=4,
            execution_horizon=12,
            schedule=schedule,
        )
        assert torch.allclose(actual, expected)


def test_rtc_action_mask_degenerate_boundary():
    mask = build_rtc_action_mask(
        total=32,
        inference_delay=16,
        execution_horizon=16,
        schedule="EXP",
    )
    assert torch.equal(mask[:16], torch.ones(16))
    assert torch.equal(mask[16:], torch.zeros(16))


def test_rtc_masks_timesteps_and_preserves_each_16d_action_vector():
    used_channels = list(range(7)) + [28] + list(range(7, 14)) + [29]
    # Robotwin layout: 16 action-vector components x 2 latent frames x
    # 16 control steps.  Values uniquely identify (time, component).
    previous = torch.empty(16, 2, 16)
    for component in range(16):
        for frame in range(2):
            for within_frame in range(16):
                time_step = frame * 16 + within_frame
                previous[component, frame, within_frame] = (
                    time_step * 100 + component
                )

    adapter = VA_PrevChunkAdapter(
        constrain_mode="RTC",
        prev_actions=previous,
        used_action_channel_ids=used_channels,
        action_num=32,
        action_dim=30,
        frame_chunk_size=2,
        action_per_frame=16,
        state_num=2,
        latent_channel=1,
        latent_height=1,
        latent_width=1,
        state_dim=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_delay=4,
        execution_horizon=12,
        rtc_attention_schedule="LINEAR",
    )

    targets = adapter.get_constrained_actions()
    weights = adapter.get_action_prefix_weights()
    assert targets.shape == (1, 30, 2, 16, 1)
    assert weights.shape == (1, 1, 2, 16, 1)

    flat_targets = targets[0, :, :, :, 0].reshape(30, 32).T
    flat_weights = weights.flatten()
    # The overlap target is the previous chunk's final H-s=20 vectors.
    for target_step in range(20):
        source_step = target_step + 12
        expected_vector = torch.zeros(30)
        for component, channel in enumerate(used_channels):
            expected_vector[channel] = source_step * 100 + component
        assert torch.equal(flat_targets[target_step], expected_vector)

    # One scalar RTC weight is broadcast over all 30 internal channels, hence
    # over all 16 effective output components of the same action timestep.
    expanded = weights.expand_as(targets)
    for time_step in range(32):
        frame, within_frame = divmod(time_step, 16)
        assert torch.unique(expanded[0, :, frame, within_frame, 0]).numel() == 1
        assert expanded[0, 0, frame, within_frame, 0] == flat_weights[time_step]


def test_rtc_preserves_14d_action_vectors_without_flattening_components():
    previous = torch.arange(14 * 2 * 16, dtype=torch.float32).reshape(14, 2, 16)
    adapter = VA_PrevChunkAdapter(
        constrain_mode="RTC",
        prev_actions=previous,
        used_action_channel_ids=list(range(14)),
        action_num=32,
        action_dim=14,
        frame_chunk_size=2,
        action_per_frame=16,
        state_num=2,
        latent_channel=1,
        latent_height=1,
        latent_width=1,
        state_dim=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_delay=4,
        execution_horizon=12,
        rtc_attention_schedule="EXP",
    )
    targets = adapter.get_constrained_actions()
    weights = adapter.get_action_prefix_weights()
    assert targets.shape == (1, 14, 2, 16, 1)
    assert weights.shape == (1, 1, 2, 16, 1)
    assert weights.expand_as(targets).shape == targets.shape
    for frame in range(2):
        for within_frame in range(16):
            assert torch.unique(
                weights.expand_as(targets)[0, :, frame, within_frame, 0]
            ).numel() == 1


def test_rtc_without_previous_action_degenerates_to_none():
    adapter = _make_mode_adapter("RTC")
    adapter_without_previous = VA_PrevChunkAdapter(
        constrain_mode="RTC",
        prev_actions=None,
        used_action_channel_ids=[0],
        action_num=8,
        action_dim=1,
        frame_chunk_size=2,
        action_per_frame=4,
        state_num=2,
        latent_channel=1,
        latent_height=1,
        latent_width=1,
        state_dim=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        inference_delay=2,
        execution_horizon=3,
    )
    assert torch.count_nonzero(adapter.get_action_prefix_weights()) > 0
    assert torch.count_nonzero(
        adapter_without_previous.get_action_prefix_weights()
    ) == 0


def test_chunk_constraint_context_exposes_live_state_only_to_fbfm():
    def make_context(mode):
        return ChunkConstraintContext(
            mode=mode,
            chunk_id=7,
            target_frame_st_id=10,
            action_targets=torch.zeros(1, 1, 2, 2, 1),
            action_mask=torch.ones(1, 1, 2, 2, 1),
            state_targets=torch.zeros(1, 1, 2, 1, 1),
            state_mask=torch.zeros(1, 1, 2, 1, 1),
        )

    for mode in ConstraintMode:
        context = make_context(mode)
        _, before, version = context.snapshot_state_constraints()
        assert version == 0
        assert torch.count_nonzero(before) == 0
        assert context.update_state_slot(
            global_slot_id=10, state=torch.tensor([[[[3.0]]]])
        )
        states, state_mask, version = context.snapshot_state_constraints()
        _, action_mask, _ = context.snapshot_action_constraints()
        assert version == 1
        assert states[0, 0, 0, 0, 0] == 3
        if mode is ConstraintMode.FBFM:
            assert state_mask[0, 0, 0, 0, 0] == 1
        else:
            assert torch.count_nonzero(state_mask) == 0
        if mode is ConstraintMode.NONE:
            assert torch.count_nonzero(action_mask) == 0
        else:
            assert torch.equal(action_mask, torch.ones_like(action_mask))

        assert not context.update_state_slot(
            global_slot_id=10, state=torch.tensor([[[[4.0]]]])
        )
        assert not context.update_state_slot(
            global_slot_id=99, state=torch.tensor([[[[4.0]]]])
        )


def test_running_solver_can_observe_feedback_added_between_steps():
    context = ChunkConstraintContext(
        mode="FBFM",
        chunk_id=1,
        target_frame_st_id=5,
        action_targets=torch.zeros(1, 1, 1, 1, 1),
        action_mask=torch.zeros(1, 1, 1, 1, 1),
        state_targets=torch.zeros(1, 1, 1, 1, 1),
        state_mask=torch.zeros(1, 1, 1, 1, 1),
    )
    targets_0, mask_0, version_0 = context.snapshot_state_constraints()
    assert version_0 == 0
    assert torch.count_nonzero(mask_0) == 0

    assert context.update_state_slot(
        global_slot_id=5, state=torch.tensor([[[[2.5]]]])
    )

    targets_1, mask_1, version_1 = context.snapshot_state_constraints()
    assert version_1 == 1
    assert targets_0[0, 0, 0, 0, 0] == 0
    assert targets_1[0, 0, 0, 0, 0] == 2.5
    assert mask_1[0, 0, 0, 0, 0] == 1


def _make_fake_feedback_server(latent_frames=1):
    from wan_va_server import VA_Server

    server = object.__new__(VA_Server)
    server.device = torch.device("cpu")
    server.dtype = torch.float32
    server.frame_st_id = 0
    server.latent_height = 1
    server.latent_width = 1
    server.job_config = SimpleNamespace(
        frame_chunk_size=2,
        feedback_obs_per_state=4,
        constraint_mode="FBFM",
    )
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
    server.feedback_last_observation_action_step = None
    server._live_feedback_queue = queue.Queue()
    server._inference_state_lock = threading.RLock()
    server._inference_running = True
    server._solver_phase = "video"
    server._cancel_requested = threading.Event()
    server.active_constraint_context = None

    encode_calls = []

    def fake_encode(obs, streaming_vae, streaming_vae_half):
        del streaming_vae, streaming_vae_half
        obs_chunk = list(obs["obs"])
        encode_calls.append(obs_chunk)
        frame_count = latent_frames(len(obs_chunk)) if callable(latent_frames) else latent_frames
        return torch.full((1, 1, frame_count, 1, 1), float(len(encode_calls)))

    server._encode_obs_with_stream_wrappers = fake_encode
    return server, encode_calls


def test_none_and_rtc_do_not_queue_or_encode_feedback():
    for mode in ("NONE", "RTC"):
        server, encode_calls = _make_fake_feedback_server()
        server.job_config.constraint_mode = mode

        result = server.enqueue_live_feedback({"feedback": True, "obs": ["frame"]})

        assert not result["feedback_queued"]
        assert result["feedback_ignored_mode"] == mode
        assert server._live_feedback_queue.empty()
        assert encode_calls == []


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


def test_feedback_accumulation_window_reopens_after_slots_are_full():
    server, _ = _make_fake_feedback_server()
    server.feedback_stream_seeded = True
    server.feedback_has_received_window = True

    server.frame_st_id = 2
    server._start_feedback_accumulation_window()
    assert server.feedback_target_frame_st_id == 2

    server._ingest_feedback_observations(["obs20", "obs24", "obs28", "obs32"], True)
    server._ingest_feedback_observations(["obs36", "obs40", "obs44", "obs48"], True)
    states, mask, count = server.feedback_state_buffer.export()
    assert count == 2
    assert torch.equal(mask, torch.tensor([1.0, 1.0]))
    assert torch.equal(states[:, 0], torch.tensor([1.0, 2.0]))

    server.frame_st_id = 3
    server._start_feedback_accumulation_window()
    states, mask, count = server.feedback_state_buffer.export()
    assert server.feedback_target_frame_st_id == 3
    assert count == 0
    assert torch.equal(mask, torch.tensor([0.0, 0.0]))
    assert torch.equal(states[:, 0], torch.tensor([0.0, 0.0]))

    server._ingest_feedback_observations(["obs52", "obs56", "obs60", "obs64"], True)
    states, mask, count = server.feedback_state_buffer.export()
    assert count == 1
    assert torch.equal(mask, torch.tensor([1.0, 0.0]))
    assert torch.equal(states[:, 0], torch.tensor([3.0, 0.0]))


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


def test_feedback_window_cutoff_keeps_history_out_of_future_state_mask():
    server, _ = _make_fake_feedback_server()
    server.feedback_stream_seeded = True
    server.feedback_has_received_window = True
    server.feedback_constraint_active = True
    server.feedback_window_start_action_step = 16

    server.feedback_pending_obs = ["obs4", "obs8", "obs12"]
    server._feedback({
        "obs": ["obs4", "obs8", "obs12", "obs16"],
        "observation_action_step": 16,
    })
    _, mask, count = server.feedback_state_buffer.export()
    assert count == 0
    assert torch.count_nonzero(mask) == 0

    server.feedback_pending_obs = ["obs17", "obs18", "obs19"]
    server._feedback({
        "obs": ["obs17", "obs18", "obs19", "obs20"],
        "observation_action_step": 20,
    })
    _, mask, count = server.feedback_state_buffer.export()
    assert count == 1
    assert torch.equal(mask, torch.tensor([1.0, 0.0]))


def test_solver_boundary_drains_feedback_into_active_context():
    server, _ = _make_fake_feedback_server()
    server.feedback_stream_seeded = True
    server.feedback_has_received_window = True
    server.feedback_constraint_active = True
    server.feedback_pending_obs = ["obs20", "obs24", "obs28"]
    server.active_constraint_context = ChunkConstraintContext(
        mode="FBFM",
        chunk_id=0,
        target_frame_st_id=0,
        action_targets=torch.zeros(1, 1, 2, 1, 1),
        action_mask=torch.zeros(1, 1, 2, 1, 1),
        state_targets=torch.zeros(1, 1, 2, 1, 1),
        state_mask=torch.zeros(1, 1, 2, 1, 1),
    )

    queued = server.enqueue_live_feedback({
        "obs": ["obs20", "obs24", "obs28", "obs32"],
        "feedback": True,
        "observation_action_step": 16,
    })
    assert queued["feedback_queued"]
    count, cancelled = server._drain_live_feedback()
    states, mask, version = (
        server.active_constraint_context.snapshot_state_constraints()
    )
    assert count == 1
    assert not cancelled
    assert version == 1
    assert states[0, 0, 0, 0, 0] == 1
    assert mask[0, 0, 0, 0, 0] == 1
    assert server.feedback_last_observation_action_step == 16


def test_static_ablation_retains_feedback_without_mutating_running_context():
    server, _ = _make_fake_feedback_server()
    server.job_config.feedback_live_enabled = False
    server.feedback_stream_seeded = True
    server.feedback_has_received_window = True
    server.feedback_constraint_active = True
    server.feedback_pending_obs = ["obs20", "obs24", "obs28"]
    server.active_constraint_context = ChunkConstraintContext(
        mode="FBFM",
        chunk_id=0,
        target_frame_st_id=0,
        action_targets=torch.zeros(1, 1, 2, 1, 1),
        action_mask=torch.zeros(1, 1, 2, 1, 1),
        state_targets=torch.zeros(1, 1, 2, 1, 1),
        state_mask=torch.zeros(1, 1, 2, 1, 1),
    )

    server._feedback({"obs": ["obs20", "obs24", "obs28", "obs32"]})

    _, context_mask, version = (
        server.active_constraint_context.snapshot_state_constraints()
    )
    _, buffered_mask, buffered_count = server.feedback_state_buffer.export()
    assert version == 0
    assert torch.count_nonzero(context_mask) == 0
    assert buffered_count == 1
    assert torch.equal(buffered_mask, torch.tensor([1.0, 0.0]))


def test_real_kv_cache_rejects_misaligned_observation_and_action_frames(
    monkeypatch, tmp_path
):
    import wan_va_server as server_module

    class FakeTransformer:
        def clear_pred_cache(self, cache_name):
            assert cache_name == "pos"

        def __call__(self, *args, **kwargs):
            raise AssertionError("misaligned history must fail before a cache write")

    server = object.__new__(server_module.VA_Server)
    server.transformer = FakeTransformer()
    server.cache_name = "pos"
    server.exp_save_root = str(tmp_path)
    server.frame_st_id = 1
    server._encode_obs = lambda obs: torch.zeros(1, 48, 1, 1, 1)
    server.preprocess_action = lambda action: torch.zeros(1, 30, 2, 16, 1)
    monkeypatch.setattr(server_module, "save_async", lambda *args, **kwargs: None)

    try:
        server._compute_kv_cache({"obs": ["frame"], "state": "actions"})
    except ValueError as exc:
        assert "time-aligned observation/action frames" in str(exc)
    else:
        raise AssertionError("expected a time-alignment error")


def test_previous_action_constraint_returns_to_normalized_model_coordinates():
    import wan_va_server as server_module

    server = object.__new__(server_module.VA_Server)
    server.job_config = SimpleNamespace(
        used_action_channel_ids=[0, 2],
        inverse_used_action_channel_ids=[0, 2, 1],
    )
    server.action_norm_method = "quantiles"
    server.actions_q01 = torch.tensor([-2.0, 0.0, 10.0]).reshape(-1, 1, 1)
    server.actions_q99 = torch.tensor([2.0, 0.0, 20.0]).reshape(-1, 1, 1)
    server.action_mask = torch.tensor([True, False, True])
    normalized = torch.tensor(
        [-0.5, 0.25, 0.0, 0.0, 0.75, -0.25]
    ).reshape(1, 3, 2, 1, 1)
    assert normalized.shape == (1, 3, 2, 1, 1)

    server.last_action = server.postprocess_action(normalized.clone())
    target = server._preprocess_previous_action_target()

    assert target.shape == (3, 2, 1)
    assert torch.allclose(target[server.action_mask], normalized[0, server.action_mask, ..., 0])
    assert torch.count_nonzero(target[~server.action_mask]) == 0
