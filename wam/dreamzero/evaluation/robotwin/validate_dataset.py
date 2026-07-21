"""Fail-fast validation for a native DreamZero RoboTwin LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "video": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
    "state": (
        "left_eef_position",
        "left_eef_rotation",
        "left_gripper",
        "right_eef_position",
        "right_eef_rotation",
        "right_gripper",
    ),
    "action": (
        "left_eef_position",
        "left_eef_rotation",
        "left_gripper",
        "right_eef_position",
        "right_eef_rotation",
        "right_gripper",
    ),
}


def validate_dataset(root: Path) -> None:
    meta = root / "meta"
    required = (
        "embodiment.json",
        "modality.json",
        "stats.json",
        "relative_stats_dreamzero.json",
        "tasks.jsonl",
        "episodes.jsonl",
    )
    for filename in required:
        if not (meta / filename).is_file():
            raise FileNotFoundError(meta / filename)
    embodiment = json.loads((meta / "embodiment.json").read_text(encoding="utf-8"))
    if embodiment.get("embodiment_tag") != "robotwin":
        raise ValueError("meta/embodiment.json must set embodiment_tag to 'robotwin'")
    modalities = json.loads((meta / "modality.json").read_text(encoding="utf-8"))
    serialized = json.dumps(modalities, sort_keys=True)
    for group, keys in EXPECTED.items():
        for key in keys:
            if key not in serialized:
                raise ValueError(f"modality.json is missing native {group} key containing {key!r}")
    stats = json.loads((meta / "relative_stats_dreamzero.json").read_text(encoding="utf-8"))
    serialized_stats = json.dumps(stats, sort_keys=True)
    for group in ("state", "action"):
        for key in EXPECTED[group]:
            if key not in serialized_stats:
                raise ValueError(f"relative_stats_dreamzero.json is missing {group}.{key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    validate_dataset(args.dataset)
    print("DreamZero RoboTwin dataset metadata validation passed")


if __name__ == "__main__":
    main()
