#!/usr/bin/env python3
"""Extract and normalize a RealSense RGB stream from a ROS2 SQLite bag."""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
OUTPUT_FPS = 24
OUTPUT_FRAMES = 168
CONTEXT_FRAMES = 48
REFERENCE_FRAMES = 121
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 704
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class CDRReader:
    def __init__(self, data: bytes) -> None:
        if data[:4] != b"\x00\x01\x00\x00":
            raise ValueError(f"expected little-endian CDR, got {data[:4].hex()}")
        self.data = data
        self.position = 4

    def align(self, size: int) -> None:
        self.position = (self.position + size - 1) // size * size

    def uint32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<I", self.data, self.position)[0]
        self.position += 4
        return value

    def int32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<i", self.data, self.position)[0]
        self.position += 4
        return value

    def uint8(self) -> int:
        value = self.data[self.position]
        self.position += 1
        return value

    def string(self) -> str:
        length = self.uint32()
        raw = self.data[self.position : self.position + length]
        self.position += length
        if not raw or raw[-1] != 0:
            raise ValueError("CDR string is missing its null terminator")
        return raw[:-1].decode("utf-8")

    def byte_sequence(self) -> memoryview:
        length = self.uint32()
        value = memoryview(self.data)[self.position : self.position + length]
        self.position += length
        if len(value) != length:
            raise ValueError("truncated CDR byte sequence")
        return value


def decode_rgb8_image(data: bytes) -> tuple[np.ndarray, dict[str, int | str]]:
    reader = CDRReader(data)
    stamp_sec = reader.int32()
    stamp_nanosec = reader.uint32()
    frame_id = reader.string()
    height = reader.uint32()
    width = reader.uint32()
    encoding = reader.string()
    is_bigendian = reader.uint8()
    step = reader.uint32()
    pixels = reader.byte_sequence()

    if encoding.lower() != "rgb8":
        raise ValueError(f"expected rgb8 image, got {encoding}")
    if is_bigendian != 0:
        raise ValueError("big-endian RGB images are not supported")
    if width != OUTPUT_WIDTH or height != 720 or step != width * 3:
        raise ValueError(f"unexpected image layout: {width}x{height}, step={step}")
    expected_bytes = height * step
    if len(pixels) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} image bytes, got {len(pixels)}")

    image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
    image = image[8 : 8 + OUTPUT_HEIGHT].copy()
    return image, {
        "stamp_sec": stamp_sec,
        "stamp_nanosec": stamp_nanosec,
        "frame_id": frame_id,
        "source_width": width,
        "source_height": height,
        "encoding": encoding,
        "step": step,
    }


def video_writer(path: Path):
    return imageio.get_writer(
        path,
        fps=OUTPUT_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-movflags", "+faststart"],
    )


def nearest_message_indices(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = timestamps[0] + np.rint(
        np.arange(OUTPUT_FRAMES, dtype=np.float64) * 1_000_000_000 / OUTPUT_FPS
    ).astype(np.int64)
    if targets[-1] > timestamps[-1]:
        raise ValueError("the bag does not contain seven seconds of color frames")

    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(targets - timestamps[left]) <= np.abs(timestamps[right] - targets)
    selected = np.where(use_left, left, right)
    return selected, targets


def save_timeline(path: Path, frames: list[np.ndarray]) -> None:
    indices = (0, 24, 48, 72, 96, 120, 144, 167)
    cell_width, cell_height, header_height = 320, 176, 48
    canvas = Image.new(
        "RGB", (cell_width * len(indices), cell_height + header_height), (18, 18, 18)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT_PATH, 22)
    for column, frame_index in enumerate(indices):
        draw.text(
            (column * cell_width + 8, 10),
            f"{frame_index / OUTPUT_FPS:.2f} s",
            fill=(245, 245, 245),
            font=font,
        )
        frame = Image.fromarray(frames[frame_index]).resize(
            (cell_width, cell_height), Image.Resampling.LANCZOS
        )
        canvas.paste(frame, (column * cell_width, header_height))
    canvas.save(path, quality=94, subsampling=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag = args.bag.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{bag}?mode=ro", uri=True)
    try:
        topic_row = connection.execute(
            "SELECT id, type, serialization_format FROM topics WHERE name = ?",
            (COLOR_TOPIC,),
        ).fetchone()
        if topic_row is None:
            raise ValueError(f"color topic not found: {COLOR_TOPIC}")
        topic_id, message_type, serialization = topic_row
        if message_type != "sensor_msgs/msg/Image" or serialization != "cdr":
            raise ValueError(
                f"unexpected color topic format: {message_type}, {serialization}"
            )
        rows = connection.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        ).fetchall()
        message_ids = np.array([row[0] for row in rows], dtype=np.int64)
        timestamps = np.array([row[1] for row in rows], dtype=np.int64)
        selected, targets = nearest_message_indices(timestamps)

        full_path = output_dir / "normalized_first_7s_24fps_1280x704.mp4"
        context_path = output_dir / "input_context_2s.mp4"
        reference_path = output_dir / "reference_future_121f.mp4"
        full_writer = video_writer(full_path)
        context_writer = video_writer(context_path)
        reference_writer = video_writer(reference_path)
        normalized_frames: list[np.ndarray] = []
        first_fields = None
        try:
            for output_index, source_index in enumerate(selected):
                message_id = int(message_ids[source_index])
                data = connection.execute(
                    "SELECT data FROM messages WHERE id = ?", (message_id,)
                ).fetchone()[0]
                frame, fields = decode_rgb8_image(data)
                if first_fields is None:
                    first_fields = fields
                normalized_frames.append(frame)
                full_writer.append_data(frame)
                if output_index < CONTEXT_FRAMES:
                    context_writer.append_data(frame)
                if output_index >= CONTEXT_FRAMES - 1:
                    reference_writer.append_data(frame)
        finally:
            full_writer.close()
            context_writer.close()
            reference_writer.close()
    finally:
        connection.close()

    if len(normalized_frames) != OUTPUT_FRAMES:
        raise RuntimeError(f"expected {OUTPUT_FRAMES} normalized frames")
    Image.fromarray(normalized_frames[CONTEXT_FRAMES - 1]).save(
        output_dir / "anchor_frame_048.png"
    )
    save_timeline(output_dir / "source_first_7s_keyframes.jpg", normalized_frames)

    sampling_errors = timestamps[selected] - targets
    metadata = {
        "source_bag": str(bag),
        "color_topic": COLOR_TOPIC,
        "source_message_count": len(timestamps),
        "source_duration_seconds": float((timestamps[-1] - timestamps[0]) / 1e9),
        "source_nominal_fps": 30,
        "source_fields": first_fields,
        "preprocessing": {
            "duration_seconds": 7.0,
            "output_fps": OUTPUT_FPS,
            "output_frame_count": OUTPUT_FRAMES,
            "vertical_crop_source_rows": [8, 711],
            "output_resolution": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            "timestamp_sampling_error_ms_max_abs": float(
                np.max(np.abs(sampling_errors)) / 1e6
            ),
            "timestamp_sampling_error_ms_mean_abs": float(
                np.mean(np.abs(sampling_errors)) / 1e6
            ),
            "reused_source_frame_count": int(len(selected) - len(np.unique(selected))),
        },
        "context": {
            "normalized_frame_range_zero_based": [0, CONTEXT_FRAMES - 1],
            "frame_count": CONTEXT_FRAMES,
            "wan_anchor_frame_zero_based": CONTEXT_FRAMES - 1,
        },
        "prediction_reference": {
            "normalized_frame_range_zero_based": [
                CONTEXT_FRAMES - 1,
                OUTPUT_FRAMES - 1,
            ],
            "frame_count_including_anchor": REFERENCE_FRAMES,
            "future_frame_count": REFERENCE_FRAMES - 1,
            "feedback_slots": (REFERENCE_FRAMES - 1) // 4,
        },
        "outputs": {
            "normalized": str(full_path),
            "context": str(context_path),
            "reference": str(reference_path),
            "anchor": str(output_dir / "anchor_frame_048.png"),
        },
    }
    (output_dir / "preprocessing.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
