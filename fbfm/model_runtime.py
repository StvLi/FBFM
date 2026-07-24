"""Single-GPU loading helpers for the RLinf DreamZero checkpoint."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict


def build_runtime_config(
    checkpoint_dir: str | Path, tokenizer_dir: str | Path
) -> Any:
    """Build an RLinf model config using local deployment paths."""
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    tokenizer = Path(tokenizer_dir).expanduser().resolve()
    with (checkpoint / "config.json").open(encoding="utf-8") as handle:
        raw = json.load(handle)
    cfg = OmegaConf.create(raw)
    with open_dict(cfg):
        cfg.model_path = str(checkpoint)
        cfg.tokenizer_path = str(tokenizer)
        cfg.metadata_json_path = str(checkpoint / "experiment_cfg" / "metadata.json")
        cfg.embodiment_tag = "libero_sim"
        cfg.precision = "bf16"
        cfg.is_lora = False
        cfg.num_action_chunks = 16
        cfg.relative_action = False
        cfg.relative_action_per_horizon = False
        cfg.relative_action_keys = []
        cfg.action_head_cfg.config.skip_component_loading = True
        cfg.action_head_cfg.config.defer_lora_injection = False
    return cfg


def build_data_transform(
    checkpoint_dir: str | Path, tokenizer_dir: str | Path
) -> Any:
    """Build the checkpoint's eval-mode LIBERO transform without loading the model."""
    from groot.vla.data.transform import ComposedModalityTransform
    from rlinf.data.datasets.dreamzero.data_transforms import (
        build_dreamzero_composed_transform,
        load_dreamzero_dataset_metadata,
    )

    cfg = build_runtime_config(checkpoint_dir, tokenizer_dir)
    transform = build_dreamzero_composed_transform(cfg, str(cfg.tokenizer_path))
    if not isinstance(transform, ComposedModalityTransform):
        raise TypeError(f"Expected ComposedModalityTransform, got {type(transform)}")
    transform.set_metadata(load_dreamzero_dataset_metadata(cfg))
    transform.eval()
    return transform


def load_policy(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    device: str = "cuda:0",
    cpu_only: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Strictly load the DreamZero policy and optionally move it to one GPU."""
    torch._dynamo.config.disable = True
    from rlinf.models.embodiment.dreamzero import get_model

    cfg = build_runtime_config(checkpoint_dir, tokenizer_dir)
    started = time.perf_counter()
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    cpu_load_seconds = time.perf_counter() - started

    model.eval()
    model.requires_grad_(False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    report: dict[str, Any] = {
        "cpu_load_seconds": cpu_load_seconds,
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "dtype": str(next(model.parameters()).dtype),
        "device": "cpu",
    }

    if not cpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the FBFM model environment")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        gpu_started = time.perf_counter()
        model.to(device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize(device)
        report.update(
            {
                "gpu_move_seconds": time.perf_counter() - gpu_started,
                "device": device,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )

    action_head = model.action_head
    action_head._device = device if not cpu_only else "cpu"
    action_head.ip_rank = 0
    action_head.ip_size = 1
    action_head.ip_group = None
    if not hasattr(action_head, "trt_engine"):
        action_head.trt_engine = None
    if not hasattr(action_head, "trt_context"):
        action_head.trt_context = None

    gc.collect()
    return model, report


def reset_policy_state(model: torch.nn.Module, seed: int) -> None:
    """Clear sequence caches when a new LIBERO episode starts."""
    action_head = model.action_head
    action_head.seed = int(seed)
    action_head.language = None
    action_head.current_start_frame = 0
    for name in (
        "kv_cache1",
        "kv_cache_neg",
        "crossattn_cache",
        "crossattn_cache_neg",
        "clip_feas",
        "ys",
    ):
        if hasattr(action_head, name):
            setattr(action_head, name, None)
