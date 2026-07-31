#!/usr/bin/env python3
"""Prepare paper-ready base/FBFM comparison material for the robot task."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 24
FUTURE_FRAME_COUNT = 121
CONTEXT_FRAME_COUNT = 48
FRAME_SIZE = (1280, 704)
KEYFRAME_INDICES = (0, 6, 12, 24, 48, 72, 96, 120)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PROMPT = (
    "Fixed high-angle RealSense video of a robotic arm positioned at the right "
    "edge of a white laboratory table. An orange miniature basketball and a "
    "black-and-white miniature soccer ball roll rapidly from left to right "
    "across the table toward the robot. As the balls approach, the robotic "
    "gripper moves inward and intercepts and stops the black-and-white soccer "
    "ball near its fingers, while the orange ball continues past and exits the "
    "scene. Realistic rigid-body physics, fixed camera, natural laboratory "
    "lighting."
)


def load_video(path: Path, expected_frames: int) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frames = np.stack([frame[..., :3] for frame in reader])
        metadata = reader.get_meta_data()
    finally:
        reader.close()
    expected_shape = (expected_frames, FRAME_SIZE[1], FRAME_SIZE[0], 3)
    if frames.shape != expected_shape:
        raise ValueError(f"unexpected video shape for {path}: {frames.shape}")
    if not math.isclose(float(metadata.get("fps", FPS)), FPS, abs_tol=0.01):
        raise ValueError(f"unexpected frame rate for {path}: {metadata.get('fps')}")
    return frames


def video_writer(path: Path):
    return imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-movflags", "+faststart"],
    )


def write_video(path: Path, frames) -> None:
    writer = video_writer(path)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def title_bar(labels: list[str], cell_width: int, height: int = 48) -> np.ndarray:
    canvas = Image.new("RGB", (cell_width * len(labels), height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 21)
    for column, label in enumerate(labels):
        draw.text(
            (column * cell_width + 12, 11),
            label,
            fill=(245, 245, 245),
            font=font,
        )
    return np.asarray(canvas)


def comparison_frames(videos: dict[str, np.ndarray]):
    labels = list(videos)
    frame_count = videos[labels[0]].shape[0]
    if any(video.shape[0] != frame_count for video in videos.values()):
        raise ValueError("comparison videos must contain the same number of frames")
    cell_width, cell_height = 480, 264
    header = title_bar(labels, cell_width)
    for frame_index in range(frame_count):
        row = np.hstack(
            [
                cv2.resize(
                    videos[label][frame_index],
                    (cell_width, cell_height),
                    interpolation=cv2.INTER_AREA,
                )
                for label in labels
            ]
        )
        yield np.vstack([header, row])


def with_context_frames(context: np.ndarray, future: np.ndarray):
    yield from context
    yield from future[1:]


def comparison_with_context_frames(
    context: np.ndarray, videos: dict[str, np.ndarray]
):
    labels = list(videos)
    cell_width, cell_height = 480, 264
    header = title_bar(labels, cell_width)
    for frame_index in range(CONTEXT_FRAME_COUNT + FUTURE_FRAME_COUNT - 1):
        if frame_index < CONTEXT_FRAME_COUNT:
            row_frames = [context[frame_index]] * len(labels)
        else:
            future_index = frame_index - CONTEXT_FRAME_COUNT + 1
            row_frames = [videos[label][future_index] for label in labels]
        row = np.hstack(
            [
                cv2.resize(
                    frame,
                    (cell_width, cell_height),
                    interpolation=cv2.INTER_AREA,
                )
                for frame in row_frames
            ]
        )
        yield np.vstack([header, row])


def save_keyframes(result_dir: Path, videos: dict[str, np.ndarray]) -> None:
    keyframe_dir = result_dir / "keyframes"
    for label, video in videos.items():
        label_dir = keyframe_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for frame_index in KEYFRAME_INDICES:
            time_ms = round(frame_index * 1000 / FPS)
            Image.fromarray(video[frame_index]).save(
                label_dir / f"t_{time_ms:04d}ms_frame_{frame_index:03d}.png"
            )


def save_keyframe_grid(path: Path, videos: dict[str, np.ndarray]) -> None:
    cell_width, cell_height, header_height = 320, 176, 48
    canvas = Image.new(
        "RGB",
        (
            cell_width * len(KEYFRAME_INDICES),
            header_height + cell_height * len(videos),
        ),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    time_font = ImageFont.truetype(FONT_PATH, 22)
    label_font = ImageFont.truetype(FONT_PATH, 20)
    for column, frame_index in enumerate(KEYFRAME_INDICES):
        draw.text(
            (column * cell_width + 8, 10),
            f"{frame_index / FPS:.2f} s",
            fill=(245, 245, 245),
            font=time_font,
        )
    for row, (label, video) in enumerate(videos.items()):
        y_position = header_height + row * cell_height
        for column, frame_index in enumerate(KEYFRAME_INDICES):
            frame = Image.fromarray(video[frame_index]).resize(
                (cell_width, cell_height), Image.Resampling.LANCZOS
            )
            canvas.paste(frame, (column * cell_width, y_position))
        draw.rectangle((4, y_position + 4, 250, y_position + 34), fill=(12, 12, 12))
        draw.text(
            (9, y_position + 6),
            label,
            fill=(255, 255, 255),
            font=label_font,
        )
    canvas.save(path, quality=95, subsampling=0)


def pixel_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    value_count = 0
    temporal_error_sum = 0.0
    temporal_value_count = 0
    for frame_index in range(FUTURE_FRAME_COUNT):
        delta = reference[frame_index].astype(np.float32)
        delta -= prediction[frame_index].astype(np.float32)
        absolute_error_sum += float(np.abs(delta).sum(dtype=np.float64))
        squared_error_sum += float(np.square(delta).sum(dtype=np.float64))
        value_count += delta.size
        if frame_index:
            reference_motion = reference[frame_index].astype(np.float32)
            reference_motion -= reference[frame_index - 1].astype(np.float32)
            prediction_motion = prediction[frame_index].astype(np.float32)
            prediction_motion -= prediction[frame_index - 1].astype(np.float32)
            temporal_delta = reference_motion - prediction_motion
            temporal_error_sum += float(
                np.abs(temporal_delta).sum(dtype=np.float64)
            )
            temporal_value_count += temporal_delta.size
    mae = absolute_error_sum / value_count
    mse = squared_error_sum / value_count
    return {
        "full_frame_mae": mae,
        "full_frame_psnr_db": 10 * math.log10(255.0**2 / max(mse, 1e-12)),
        "temporal_gradient_mae": temporal_error_sum / temporal_value_count,
    }


def audit_summary(path: Path) -> dict:
    events = json.loads(path.read_text(encoding="utf-8"))
    begin = next(event for event in events if event["event"] == "generation_begin")
    end = next(event for event in events if event["event"] == "generation_end")
    updates = [event for event in events if event["event"] == "feedback_update"]
    guidance = [event for event in events if event["event"] == "solver_guidance"]
    guided = [event for event in guidance if event.get("guided") is True]
    return {
        "generation": begin,
        "feedback_update_count": len(updates),
        "feedback_slots": [event["slot"] for event in updates],
        "feedback_release_steps": [event["release_step"] for event in updates],
        "guided_solver_step_count": len(guided),
        "unguided_solver_step_count": len(guidance) - len(guided),
        "generation_end": end,
    }


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "results" / "robot_arm_ball_stop"
    paths = {
        "reference": result_dir / "reference_future_121f.mp4",
        "wan2.2_base": result_dir / "base_future.mp4",
        "fbfm_ours": result_dir / "fbfm_ours_future.mp4",
    }
    videos = {
        label: load_video(path, FUTURE_FRAME_COUNT) for label, path in paths.items()
    }
    display_videos = {
        "Reference": videos["reference"],
        "Wan2.2 Base": videos["wan2.2_base"],
        "FBFM Ours": videos["fbfm_ours"],
    }

    write_video(
        result_dir / "reference_base_fbfm_future.mp4",
        comparison_frames(display_videos),
    )
    save_keyframe_grid(
        result_dir / "reference_base_fbfm_keyframes.jpg",
        display_videos,
    )
    save_keyframes(result_dir, videos)

    context = load_video(result_dir / "input_context_2s.mp4", CONTEXT_FRAME_COUNT)
    write_video(
        result_dir / "base_with_context.mp4",
        with_context_frames(context, videos["wan2.2_base"]),
    )
    write_video(
        result_dir / "fbfm_ours_with_context.mp4",
        with_context_frames(context, videos["fbfm_ours"]),
    )
    write_video(
        result_dir / "reference_base_fbfm_with_context.mp4",
        comparison_with_context_frames(context, display_videos),
    )

    summary = {
        "experiment": "RealSense D435i robot-arm ball-stopping prediction",
        "prompt": PROMPT,
        "timing": {
            "fps": FPS,
            "context_frames": CONTEXT_FRAME_COUNT,
            "context_duration_seconds": 2.0,
            "prediction_frames_including_anchor": FUTURE_FRAME_COUNT,
            "prediction_duration_seconds": 5.0,
            "keyframe_indices": list(KEYFRAME_INDICES),
            "keyframe_times_seconds": [index / FPS for index in KEYFRAME_INDICES],
        },
        "common_generation_config": {
            "model": "Wan2.2-TI2V-5B",
            "resolution": list(FRAME_SIZE),
            "seed": 0,
            "sampling_steps": 50,
            "sample_shift": 5.0,
            "cfg_scale": 5.0,
        },
        "fbfm_config": {
            "feedback_slots": 30,
            "observed_future_frames": 120,
            "observations_per_slot": 4,
            "state_weight": 1.0,
            "kp": 1.0,
            "beta": 10.0,
        },
        "pixel_metrics_against_reference": {
            "wan2.2_base": pixel_metrics(
                videos["reference"], videos["wan2.2_base"]
            ),
            "fbfm_ours": pixel_metrics(videos["reference"], videos["fbfm_ours"]),
        },
        "audits": {
            "wan2.2_base": audit_summary(result_dir / "base_future.json"),
            "fbfm_ours": audit_summary(result_dir / "fbfm_ours_future.json"),
        },
        "artifacts": {
            "source_bag": str(result_dir.parents[1] / "examples" / "recording.db3"),
            "normalized_first_7s": str(
                result_dir / "normalized_first_7s_24fps_1280x704.mp4"
            ),
            "context": str(result_dir / "input_context_2s.mp4"),
            "reference_future": str(paths["reference"]),
            "base_future": str(paths["wan2.2_base"]),
            "fbfm_future": str(paths["fbfm_ours"]),
            "future_comparison": str(
                result_dir / "reference_base_fbfm_future.mp4"
            ),
            "comparison_with_context": str(
                result_dir / "reference_base_fbfm_with_context.mp4"
            ),
            "keyframe_grid": str(
                result_dir / "reference_base_fbfm_keyframes.jpg"
            ),
            "individual_keyframes": str(result_dir / "keyframes"),
        },
    }
    (result_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
