"""Single-GPU loading helpers for the RLinf DreamZero checkpoint."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

WAN_TEXT_ENCODER_FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"
WAN_VAE_FILENAME = "Wan2.2_VAE.pth"


def _local_dreamzero_components(
    wan_checkpoint_dir: str | Path,
    image_encoder_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Resolve and validate every external DreamZero model component."""
    wan_checkpoint = Path(wan_checkpoint_dir).expanduser().resolve()
    text_encoder = wan_checkpoint / WAN_TEXT_ENCODER_FILENAME
    vae = wan_checkpoint / WAN_VAE_FILENAME
    image_encoder = Path(image_encoder_path).expanduser().resolve()

    missing: list[tuple[str, Path]] = []
    if not wan_checkpoint.is_dir():
        missing.append(("Wan2.2 checkpoint directory", wan_checkpoint))
    if not text_encoder.is_file():
        missing.append(("Wan text encoder", text_encoder))
    if not vae.is_file():
        missing.append(("Wan2.2 VAE", vae))
    if not image_encoder.is_file():
        missing.append(("CLIP image encoder", image_encoder))
    if missing:
        details = "\n".join(f"  - {label}: {path}" for label, path in missing)
        raise FileNotFoundError(
            "DreamZero local-component preflight failed; refusing an implicit "
            f"model download. Missing:\n{details}"
        )
    return wan_checkpoint, text_encoder, image_encoder, vae


def build_runtime_config(
    checkpoint_dir: str | Path,
    tokenizer_dir: str | Path,
    *,
    wan_checkpoint_dir: str | Path | None = None,
    image_encoder_path: str | Path | None = None,
) -> Any:
    """Build an RLinf model config using local deployment paths.

    The component arguments remain optional for metadata-only callers. Model
    loading supplies both arguments so stale training-machine paths can never
    trigger an implicit multi-gigabyte download.
    """
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    tokenizer = Path(tokenizer_dir).expanduser().resolve()
    if (wan_checkpoint_dir is None) != (image_encoder_path is None):
        raise ValueError(
            "wan_checkpoint_dir and image_encoder_path must be provided together"
        )
    components = (
        _local_dreamzero_components(wan_checkpoint_dir, image_encoder_path)
        if wan_checkpoint_dir is not None and image_encoder_path is not None
        else None
    )
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
        if components is not None:
            wan_checkpoint, text_encoder, image_encoder, vae = components
            cfg.diffusion_model_pretrained_path = str(wan_checkpoint)
            cfg.text_encoder_pretrained_path = str(text_encoder)
            cfg.image_encoder_pretrained_path = str(image_encoder)
            cfg.vae_pretrained_path = str(vae)

            action_config = cfg.action_head_cfg.config
            action_config.diffusion_model_cfg.diffusion_model_pretrained_path = str(
                wan_checkpoint
            )
            action_config.text_encoder_cfg.text_encoder_pretrained_path = str(
                text_encoder
            )
            action_config.image_encoder_cfg.image_encoder_pretrained_path = str(
                image_encoder
            )
            action_config.vae_cfg.vae_pretrained_path = str(vae)
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
    wan_checkpoint_dir: str | Path,
    image_encoder_path: str | Path,
    device: str = "cuda:0",
    cpu_only: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Strictly load the DreamZero policy and optionally move it to one GPU."""
    cfg = build_runtime_config(
        checkpoint_dir,
        tokenizer_dir,
        wan_checkpoint_dir=wan_checkpoint_dir,
        image_encoder_path=image_encoder_path,
    )
    torch._dynamo.config.disable = True
    from rlinf.models.embodiment.dreamzero import get_model

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
