"""Run paired direct/FBFM visual prediction with Wan2.2 TI2V-5B."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import imageio.v3 as iio
import torch
from PIL import Image
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.fbfm.feedback import (
    DEFAULT_KP,
    DEFAULT_STATE_WEIGHT,
    FeedbackMode,
    evenly_spaced_release_steps,
)
from wan.fbfm.pipeline import WanTI2VFBFM
from wan.utils.utils import save_video

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)


def load_frame_sequence(path: str | Path) -> list[Image.Image]:
    """Load a video or lexically ordered image directory into RGB frames."""
    source = Path(path)
    if source.is_dir():
        files = sorted(
            item for item in source.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise ValueError(f"no images found in {source}")
        return [Image.open(item).convert("RGB") for item in files]
    if not source.is_file():
        raise FileNotFoundError(source)
    return [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(source)]


def parse_release_steps(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    steps = [int(item.strip()) for item in value.split(",")]
    if steps != sorted(steps):
        raise ValueError("feedback release steps must be sorted")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wan2.2 visual-only FBFM state-feedback inference"
    )
    parser.add_argument(
        "--ckpt-dir",
        default=os.environ.get("WAN22_CKPT_DIR"),
    )
    parser.add_argument("--mode", choices=["DIRECT", "FBFM"], required=True)
    parser.add_argument(
        "--image", help="Launch image; defaults to feedback video frame 0"
    )
    parser.add_argument(
        "--feedback-video",
        help="Ground-truth video or frame directory including the launch frame",
    )
    parser.add_argument(
        "--feedback-release-steps",
        help="Comma-separated solver steps, one for each four-frame feedback group",
    )
    parser.add_argument(
        "--feedback-slots",
        type=int,
        default=1,
        help="Number of evenly released slots when explicit release steps are omitted",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--size", choices=["1280*704", "704*1280"], default="1280*704")
    parser.add_argument(
        "--max-area",
        type=int,
        help="Override output pixel area for integration tests; paper runs should use --size",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        help="Defaults to the official checkpoint configuration",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        help="Defaults to the official checkpoint configuration",
    )
    parser.add_argument(
        "--sample-shift",
        type=float,
        help="Defaults to the official checkpoint configuration",
    )
    parser.add_argument(
        "--guide-scale",
        type=float,
        help="Defaults to the official checkpoint configuration",
    )
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument(
        "--kp",
        type=float,
        default=DEFAULT_KP,
        help="Gain applied to the complete FBFM velocity correction",
    )
    parser.add_argument(
        "--state-weight",
        type=float,
        default=DEFAULT_STATE_WEIGHT,
        help="Visual-state residual scale; defaults to 1.0 for state-only FBFM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--t5-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offload-model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--convert-model-dtype",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint DiT blocks during FBFM VJPs to reduce CUDA memory",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic CUDA algorithms for paired paper comparisons",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", help="JSON audit path; defaults beside output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt_dir:
        raise ValueError("provide --ckpt-dir or set WAN22_CKPT_DIR")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    config = WAN_CONFIGS["ti2v-5B"]
    frame_num = args.frame_num if args.frame_num is not None else config.frame_num
    sample_steps = (
        args.sample_steps if args.sample_steps is not None else config.sample_steps
    )
    sample_shift = (
        args.sample_shift if args.sample_shift is not None else config.sample_shift
    )
    guide_scale = (
        args.guide_scale if args.guide_scale is not None else config.sample_guide_scale
    )
    mode = FeedbackMode.parse(args.mode)
    sequence = load_frame_sequence(args.feedback_video) if args.feedback_video else []
    if args.image:
        anchor = Image.open(args.image).convert("RGB")
    elif sequence:
        anchor = sequence[0]
    else:
        raise ValueError("provide --image or --feedback-video")

    future_frames = sequence[1:] if sequence else []
    release_steps = parse_release_steps(args.feedback_release_steps)
    if mode is FeedbackMode.FBFM and not release_steps:
        release_steps = evenly_spaced_release_steps(args.feedback_slots, sample_steps)
    if mode is FeedbackMode.FBFM and not sequence:
        raise ValueError("FBFM mode requires --feedback-video")

    pipeline = WanTI2VFBFM(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device,
        rank=0,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )
    result = pipeline.generate_with_feedback(
        input_prompt=args.prompt,
        image=anchor,
        mode=mode,
        feedback_frames=future_frames,
        feedback_release_steps=release_steps,
        max_area=args.max_area or MAX_AREA_CONFIGS[args.size],
        frame_num=frame_num,
        shift=sample_shift,
        sampling_steps=sample_steps,
        guide_scale=guide_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=args.offload_model,
        beta=args.beta,
        state_weight=args.state_weight,
        kp=args.kp,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if result.video is None:
        raise RuntimeError("rank 0 did not receive a decoded video")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(
        result.video.unsqueeze(0),
        save_file=str(output_path),
        fps=config.sample_fps,
    )
    audit_path = Path(args.audit) if args.audit else output_path.with_suffix(".json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(result.audit, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    LOGGER.info("Saved video to %s", output_path)
    LOGGER.info("Saved FBFM audit to %s", audit_path)


if __name__ == "__main__":
    main()
