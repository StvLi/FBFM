import inspect
import pathlib
import sys
import types
from dataclasses import dataclass
from enum import Enum

import torch


DREAMZERO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DREAMZERO_ROOT))

from groot.vla.model.dreamzero.modules.fbfm_guidance import (  # noqa: E402
    ConstraintMode,
    FBFMGuidanceConfig,
    JointConstraints,
    apply_joint_guidance,
)


def _install_minimal_diffusers(monkeypatch):
    """Install only the interfaces used by the vendored FlowUniPC module."""

    try:
        import diffusers  # noqa: F401

        return
    except ImportError:
        pass

    diffusers = types.ModuleType("diffusers")
    configuration = types.ModuleType("diffusers.configuration_utils")
    schedulers = types.ModuleType("diffusers.schedulers")
    scheduling = types.ModuleType("diffusers.schedulers.scheduling_utils")

    class ConfigMixin:
        def register_to_config(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self.config, key, value)

    def register_to_config(initializer):
        signature = inspect.signature(initializer)

        def wrapped(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            values = {key: value for key, value in bound.arguments.items() if key != "self"}
            self.config = types.SimpleNamespace(**values)
            return initializer(self, *args, **kwargs)

        return wrapped

    class KarrasDiffusionSchedulers(Enum):
        FAKE = "fake"

    class SchedulerMixin:
        pass

    @dataclass
    class SchedulerOutput:
        prev_sample: torch.Tensor

    configuration.ConfigMixin = ConfigMixin
    configuration.register_to_config = register_to_config
    scheduling.KarrasDiffusionSchedulers = KarrasDiffusionSchedulers
    scheduling.SchedulerMixin = SchedulerMixin
    scheduling.SchedulerOutput = SchedulerOutput
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.configuration_utils", configuration)
    monkeypatch.setitem(sys.modules, "diffusers.schedulers", schedulers)
    monkeypatch.setitem(sys.modules, "diffusers.schedulers.scheduling_utils", scheduling)


def _redirect_cuda_tensor_to_cpu(monkeypatch):
    original = torch.Tensor.to

    def redirected(tensor, *args, **kwargs):
        args = list(args)
        if args and str(args[0]).startswith("cuda"):
            args[0] = torch.device("cpu")
        if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
            kwargs["device"] = torch.device("cpu")
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", redirected)


def test_none_mode_matches_unmodified_unipc_step_by_step(monkeypatch):
    _install_minimal_diffusers(monkeypatch)
    _redirect_cuda_tensor_to_cpu(monkeypatch)
    from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler

    baseline_video = FlowUniPCMultistepScheduler(shift=1.0)
    none_video = FlowUniPCMultistepScheduler(shift=1.0)
    baseline_action = FlowUniPCMultistepScheduler(shift=1.0)
    none_action = FlowUniPCMultistepScheduler(shift=1.0)
    schedulers = (baseline_video, none_video, baseline_action, none_action)
    for scheduler in schedulers:
        for name in ("multistep_uni_p_bh_update", "multistep_uni_c_bh_update"):
            compiled = getattr(scheduler, name)
            original = getattr(compiled, "_torchdynamo_orig_callable", None)
            if original is not None:
                setattr(scheduler, name, types.MethodType(original, scheduler))
        scheduler.set_timesteps(4, device="cpu")

    torch.manual_seed(7)
    baseline_video_sample = torch.randn(1, 2, 1, 2, 2)
    none_video_sample = baseline_video_sample.clone()
    baseline_action_sample = torch.randn(1, 3, 2)
    none_action_sample = baseline_action_sample.clone()
    config = FBFMGuidanceConfig(mode=ConstraintMode.NONE)

    for index in range(4):
        video_flow = 0.2 * baseline_video_sample + index / 100
        action_flow = -0.1 * baseline_action_sample + index / 200
        none_video_flow, none_action_flow = apply_joint_guidance(
            video_sample=none_video_sample,
            action_sample=none_action_sample,
            video_flow=video_flow.clone(),
            action_flow=action_flow.clone(),
            video_local_flow=video_flow.clone(),
            action_local_flow=action_flow.clone(),
            video_sigma=none_video.sigmas[index],
            action_sigma=none_action.sigmas[index],
            constraints=JointConstraints(),
            config=config,
        )
        baseline_video_sample = baseline_video.step(
            video_flow,
            baseline_video.timesteps[index],
            baseline_video_sample,
            step_index=index,
            return_dict=False,
        )[0]
        none_video_sample = none_video.step(
            none_video_flow,
            none_video.timesteps[index],
            none_video_sample,
            step_index=index,
            return_dict=False,
        )[0]
        baseline_action_sample = baseline_action.step(
            action_flow,
            baseline_action.timesteps[index],
            baseline_action_sample,
            step_index=index,
            return_dict=False,
        )[0]
        none_action_sample = none_action.step(
            none_action_flow,
            none_action.timesteps[index],
            none_action_sample,
            step_index=index,
            return_dict=False,
        )[0]
        torch.testing.assert_close(none_video_sample, baseline_video_sample, rtol=0, atol=0)
        torch.testing.assert_close(none_action_sample, baseline_action_sample, rtol=0, atol=0)
        assert none_video.lower_order_nums == baseline_video.lower_order_nums
        assert none_action.lower_order_nums == baseline_action.lower_order_nums
        torch.testing.assert_close(none_video.last_sample, baseline_video.last_sample, rtol=0, atol=0)
        torch.testing.assert_close(none_action.last_sample, baseline_action.last_sample, rtol=0, atol=0)
