#!/usr/bin/env python3
"""Prepare visual and numeric material for the ball kp/slot ablation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 24
FRAME_COUNT = 121
FRAME_SIZE = (1280, 704)
KEYFRAME_INDICES = (0, 6, 12, 24, 48, 72, 96, 120)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_video(path: Path, expected_frames: int = FRAME_COUNT) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        frames = np.stack([frame[..., :3] for frame in reader])
        metadata = reader.get_meta_data()
    finally:
        reader.close()
    if frames.shape != (expected_frames, FRAME_SIZE[1], FRAME_SIZE[0], 3):
        raise ValueError(f"unexpected video shape for {path}: {frames.shape}")
    if not math.isclose(float(metadata.get("fps", FPS)), FPS, abs_tol=0.01):
        raise ValueError(f"unexpected frame rate for {path}: {metadata.get('fps')}")
    return frames


def write_video(path: Path, frames, *, fps: int = FPS) -> None:
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def header(labels: list[str], cell_width: int, height: int = 48) -> np.ndarray:
    canvas = Image.new("RGB", (cell_width * len(labels), height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 21)
    for index, label in enumerate(labels):
        draw.text((index * cell_width + 12, 11), label, fill=(245, 245, 245), font=font)
    return np.asarray(canvas)


def comparison_frames(videos: dict[str, np.ndarray]):
    labels = list(videos)
    frame_count = videos[labels[0]].shape[0]
    if any(video.shape[0] != frame_count for video in videos.values()):
        raise ValueError("comparison videos must have the same frame count")
    top = header(labels, 384)
    for frame_index in range(frame_count):
        row = np.hstack(
            [
                cv2.resize(
                    videos[label][frame_index],
                    (384, 212),
                    interpolation=cv2.INTER_AREA,
                )
                for label in labels
            ]
        )
        yield np.vstack([top, row])


def save_keyframe_grid(path: Path, videos: dict[str, np.ndarray]) -> None:
    cell_width, cell_height, header_height = 320, 176, 48
    labels = list(videos)
    canvas = Image.new(
        "RGB",
        (cell_width * len(KEYFRAME_INDICES), header_height + cell_height * len(labels)),
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
        y = header_height + row * cell_height
        for column, frame_index in enumerate(KEYFRAME_INDICES):
            frame = Image.fromarray(video[frame_index]).resize(
                (cell_width, cell_height), Image.Resampling.LANCZOS
            )
            canvas.paste(frame, (column * cell_width, y))
        draw.rectangle((4, y + 4, 250, y + 33), fill=(12, 12, 12))
        draw.text((9, y + 6), label, fill=(255, 255, 255), font=label_font)
    canvas.save(path, quality=94, subsampling=0)


def table_mask() -> np.ndarray:
    mask = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.uint8)
    polygon = np.array(
        [(0, 341), (774, 111), (1279, 358), (1279, 541), (263, 703), (0, 550)],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def frame_metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray):
    delta = reference.astype(np.float32) - prediction.astype(np.float32)
    roi = delta[:, mask]
    mae = np.mean(np.abs(roi), axis=(1, 2))
    mse = np.mean(np.square(roi), axis=(1, 2))
    psnr = 10 * np.log10((255.0**2) / np.maximum(mse, 1e-12))
    return mae, psnr


def orange_center(frame: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    orange = cv2.inRange(hsv, np.array([0, 120, 50]), np.array([20, 255, 255]))
    orange[:240] = 0
    orange[:, :500] = 0
    orange[:, 950:] = 0
    count, _, stats, centers = cv2.connectedComponentsWithStats(orange)
    candidates = [
        index for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 500
    ]
    if not candidates:
        return None
    selected = max(candidates, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    return centers[selected]


def trajectory_summary(
    reference: np.ndarray, prediction: np.ndarray
) -> dict[str, float | int]:
    reference_centers = [orange_center(frame) for frame in reference[:76]]
    predicted_centers = [orange_center(frame) for frame in prediction[:76]]
    errors = [
        float(np.linalg.norm(reference_center - predicted_center))
        for reference_center, predicted_center in zip(
            reference_centers, predicted_centers, strict=True
        )
        if reference_center is not None and predicted_center is not None
    ]
    return {
        "detected_frames": len(errors),
        "mean_center_error_px": float(np.mean(errors)) if errors else math.nan,
        "median_center_error_px": float(np.median(errors)) if errors else math.nan,
    }


def audit_summary(path: Path) -> dict[str, float | int | list[float]]:
    events = json.loads(path.read_text(encoding="utf-8"))
    guided = [event for event in events if event.get("guided") is True]
    effective_ratios = [
        event["kp"]
        * event["guidance_weight"]
        * event["correction_norm"]
        / event["base_velocity_norm"]
        for event in guided
    ]
    guided_ratios = [
        event["guided_velocity_norm"] / event["base_velocity_norm"] for event in guided
    ]
    return {
        "guided_steps": len(guided),
        "feedback_updates": sum(
            event.get("event") == "feedback_update" for event in events
        ),
        "kp_values": sorted({event["kp"] for event in guided}),
        "effective_correction_to_base_mean": float(np.mean(effective_ratios)),
        "effective_correction_to_base_max": float(np.max(effective_ratios)),
        "guided_to_base_velocity_mean": float(np.mean(guided_ratios)),
        "guided_to_base_velocity_max": float(np.max(guided_ratios)),
    }


def main() -> None:
    result_dir = Path(__file__).resolve().parents[1] / "results" / "ball_meet_ball"
    paths = {
        "REFERENCE": result_dir / "reference_future_121f.mp4",
        "BASE (0)": result_dir / "direct_future.mp4",
        "FBFM 10, kp=.05": result_dir / "fbfm_kp_0p05_10slots_future.mp4",
        "FBFM 20, kp=.05": result_dir / "fbfm_kp_0p05_20slots_future.mp4",
        "FBFM 30, kp=.05": result_dir / "fbfm_kp_0p05_30slots_future.mp4",
    }
    videos = {label: load_video(path) for label, path in paths.items()}

    write_video(
        result_dir / "reference_base_fbfm_kp_0p05_10_20_30_future.mp4",
        comparison_frames(videos),
    )
    save_keyframe_grid(
        result_dir / "reference_base_fbfm_kp_0p05_10_20_30_keyframes.jpg",
        videos,
    )

    context = load_video(result_dir / "input_context_2s.mp4", expected_frames=48)
    with_context = {
        label: np.concatenate([context, video[1:]], axis=0)
        for label, video in videos.items()
    }
    for slots in (10, 20, 30):
        label = f"FBFM {slots}, kp=.05"
        write_video(
            result_dir / f"fbfm_kp_0p05_{slots}slots_with_context.mp4",
            with_context[label],
        )
    write_video(
        result_dir / "reference_base_fbfm_kp_0p05_10_20_30_with_context.mp4",
        comparison_frames(with_context),
    )

    reference = videos["REFERENCE"]
    mask = table_mask()
    metrics = {}
    frame_values = {}
    for label, video in videos.items():
        if label == "REFERENCE":
            continue
        mae, psnr = frame_metrics(reference, video, mask)
        frame_values[label] = (mae, psnr)
        metrics[label] = {
            "overall_table_roi_mae": float(np.mean(mae)),
            "overall_table_roi_psnr_db": float(np.mean(psnr)),
            "segments_table_roi_mae": {
                "0-40": float(np.mean(mae[:41])),
                "41-80": float(np.mean(mae[41:81])),
                "81-120": float(np.mean(mae[81:])),
            },
            "orange_ball_frames_0_to_75": trajectory_summary(reference, video),
        }

    kp_1_metrics = {}
    kp_1_paths = {
        "FBFM 10, kp=1": result_dir / "fbfm_state_weight_1_10slots_future.mp4",
        "FBFM 20, kp=1": result_dir / "fbfm_state_weight_1_20slots_future.mp4",
        "FBFM 30, kp=1": result_dir / "fbfm_state_weight_1_future.mp4",
    }
    for label, path in kp_1_paths.items():
        video = load_video(path)
        mae, psnr = frame_metrics(reference, video, mask)
        kp_1_metrics[label] = {
            "overall_table_roi_mae": float(np.mean(mae)),
            "overall_table_roi_psnr_db": float(np.mean(psnr)),
            "segments_table_roi_mae": {
                "0-40": float(np.mean(mae[:41])),
                "41-80": float(np.mean(mae[41:81])),
                "81-120": float(np.mean(mae[81:])),
            },
            "orange_ball_frames_0_to_75": trajectory_summary(reference, video),
        }

    summary = {
        "common": {
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "resolution": list(FRAME_SIZE),
            "seed": 0,
            "sampling_steps": 50,
            "sample_shift": 5.0,
            "cfg_scale": 5.0,
            "beta": 10.0,
            "state_weight": 1.0,
            "kp": 0.05,
        },
        "metrics": metrics,
        "kp_1_metrics_same_evaluator": kp_1_metrics,
        "audits": {
            f"FBFM {slots}, kp=.05": audit_summary(
                result_dir / f"fbfm_kp_0p05_{slots}slots_future.json"
            )
            for slots in (10, 20, 30)
        },
        "table_roi_polygon": [
            [0, 341],
            [774, 111],
            [1279, 358],
            [1279, 541],
            [263, 703],
            [0, 550],
        ],
        "orange_detector": {
            "opencv_hsv_range": [[0, 120, 50], [20, 255, 255]],
            "search_bounds": {"x": [500, 950], "y": [240, 704]},
            "minimum_component_area_px": 500,
        },
    }
    (result_dir / "kp_0p05_slot_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    csv_path = result_dir / "kp_0p05_slot_ablation_frame_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["frame_index", "time_s"]
        for label in frame_values:
            key = label.lower().replace(" ", "_").replace(",", "")
            fieldnames.extend([f"{key}_table_roi_mae", f"{key}_table_roi_psnr_db"])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in range(FRAME_COUNT):
            row = {"frame_index": frame_index, "time_s": frame_index / FPS}
            for label, (mae, psnr) in frame_values.items():
                key = label.lower().replace(" ", "_").replace(",", "")
                row[f"{key}_table_roi_mae"] = float(mae[frame_index])
                row[f"{key}_table_roi_psnr_db"] = float(psnr[frame_index])
            writer.writerow(row)


if __name__ == "__main__":
    main()
