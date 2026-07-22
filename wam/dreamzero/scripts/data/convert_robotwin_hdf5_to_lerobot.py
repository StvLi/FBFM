"""Convert native RoboTwin HDF5 demonstrations to a LeRobot v2 dataset.

The raw trajectory records state after every expert control target.  Following
RoboTwin's official ACT/pi0 preprocessing, observation is frame ``t`` and the
supervised action is frame ``t + 1``.  Both are represented as absolute dual
EEF XYZ/Euler/gripper vectors; joint positions are deliberately not relabelled
as EEF state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

DREAMZERO_ROOT = Path(__file__).resolve().parents[2]
if str(DREAMZERO_ROOT) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_ROOT))

from evaluation.robotwin.representation import pack_eef14


CAMERAS = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


def _episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)\.hdf5", path.name)
    if match is None:
        raise ValueError(f"unexpected RoboTwin episode filename {path.name!r}")
    return int(match.group(1))


def _decode_rgb(value: bytes | np.bytes_) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode a RoboTwin RGB frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _instruction(raw_root: Path, episode_index: int) -> str:
    path = raw_root / "instructions" / f"episode{episode_index}.json"
    if not path.is_file():
        return "Pick up the target bottle and keep it upright."
    payload = json.loads(path.read_text(encoding="utf-8"))
    for split in ("seen", "unseen"):
        values = payload.get(split, [])
        if values:
            text = str(values[0]).strip()
            if text:
                return text
    raise ValueError(f"{path} contains no usable instruction")


def load_episode(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, list[np.ndarray]]]:
    """Return aligned state[t], action[t+1] and RGB[t] arrays."""

    with h5py.File(path, "r") as handle:
        eef = pack_eef14(
            handle["endpose/left_endpose"][:],
            handle["endpose/left_gripper"][:],
            handle["endpose/right_endpose"][:],
            handle["endpose/right_gripper"][:],
        )
        if eef.shape[0] < 2:
            raise ValueError(f"{path} needs at least two frames")
        images = {
            target: [_decode_rgb(value) for value in handle[f"observation/{source}/rgb"][:-1]]
            for source, target in CAMERAS.items()
        }
    expected = eef.shape[0] - 1
    if any(len(frames) != expected for frames in images.values()):
        raise ValueError(f"{path} camera/state length mismatch")
    return eef[:-1], eef[1:], images


def _copy_task_index_to_annotation(output_root: Path) -> None:
    """Expose LeRobot's resolved language-task index to DreamZero.

    LeRobot assigns ``task_index`` while saving each episode, so the correct
    value is not available when a frame is first added.  Copying that column
    after all episodes are finalized preserves repeated instructions and does
    not rely on episode ordering.
    """

    parquet_paths = sorted((output_root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files under {output_root / 'data'}")
    for path in parquet_paths:
        frame = pd.read_parquet(path)
        if "task_index" not in frame:
            raise KeyError(f"{path} has no task_index column")
        frame["annotation.task"] = frame["task_index"].astype(np.int64)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["annotation.task"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")


def convert(raw_root: Path, output_root: Path, *, repo_id: str, fps: int) -> Path:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted((raw_root / "data").glob("episode*.hdf5"), key=_episode_index)
    if not episodes:
        raise FileNotFoundError(f"no episode*.hdf5 files under {raw_root / 'data'}")
    first_state, _, first_images = load_episode(episodes[0])
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["eef_state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["eef_action"],
        },
    }
    del first_state
    for camera, frames in first_images.items():
        height, width, channels = frames[0].shape
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset root {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type="robotwin_aloha_agilex",
        features=features,
        use_videos=True,
        image_writer_processes=8,
        image_writer_threads=4,
    )
    for episode_path in episodes:
        episode_index = _episode_index(episode_path)
        state, action, images = load_episode(episode_path)
        task = _instruction(raw_root, episode_index)
        for frame_index in range(state.shape[0]):
            frame = {
                "observation.state": state[frame_index],
                "action": action[frame_index],
                "task": task,
            }
            for camera, values in images.items():
                frame[f"observation.images.{camera}"] = values[frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
    del dataset
    _copy_task_index_to_annotation(output_root)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/robotwin-adjust-bottle-clean50")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    result = convert(args.raw_root, args.output_root, repo_id=args.repo_id, fps=args.fps)
    print(result)


if __name__ == "__main__":
    main()
