"""Fail-fast validation for a native DreamZero RoboTwin LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
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

RELATIVE_ACTION_KEYS = ("left_eef_position", "right_eef_position")


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
    packed_stats = json.loads((meta / "stats.json").read_text(encoding="utf-8"))
    for original_key in ("observation.state", "action"):
        if original_key not in packed_stats:
            raise ValueError(f"stats.json is missing packed feature {original_key!r}")
        for statistic in ("mean", "std", "min", "max", "q01", "q99"):
            values = packed_stats[original_key].get(statistic)
            if not isinstance(values, list) or len(values) != 14:
                raise ValueError(f"stats.json {original_key}.{statistic} must contain 14 values")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"stats.json {original_key}.{statistic} contains non-finite values")

    relative_stats = json.loads((meta / "relative_stats_dreamzero.json").read_text(encoding="utf-8"))
    unexpected = sorted(set(relative_stats) - set(RELATIVE_ACTION_KEYS))
    if unexpected:
        raise ValueError(f"relative_stats_dreamzero.json has non-positional relative keys: {unexpected}")
    for key in RELATIVE_ACTION_KEYS:
        if key not in relative_stats:
            raise ValueError(f"relative_stats_dreamzero.json is missing {key}")
        for statistic in ("mean", "std", "min", "max", "q01", "q99"):
            values = relative_stats[key].get(statistic)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(f"relative_stats_dreamzero.json {key}.{statistic} must contain 3 values")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"relative_stats_dreamzero.json {key}.{statistic} contains non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    validate_dataset(args.dataset)
    print("DreamZero RoboTwin dataset metadata validation passed")


if __name__ == "__main__":
    main()
