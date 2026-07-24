#!/usr/bin/env python3
"""Aggregate the resumable 50-task RoboTwin FBFM benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path


TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - half_width, center + half_width]


def video_episodes(video_root: Path, seed_start: int) -> list[dict]:
    episodes = []
    for video in video_root.glob("*.mp4"):
        match = re.match(r"(\d+)_.*_(True|False)\.mp4$", video.name)
        if match:
            episodes.append(
                {
                    "seed": seed_start + int(match.group(1)),
                    "success": match.group(2) == "True",
                    "video": str(video.resolve()),
                }
            )
    return sorted(episodes, key=lambda item: item["seed"])


def local_task_record(root: Path, task: str, requested: int) -> dict:
    task_root = root / "tasks" / task
    result_paths = list(task_root.glob(f"client/stseed-*/metrics/{task}/res.json"))
    episodes = []
    result = None
    if len(result_paths) == 1:
        result = json.loads(result_paths[0].read_text(encoding="utf-8"))
        seed_text = result_paths[0].parts[-4]
        seed_start = int(seed_text.removeprefix("stseed-"))
        video_root = result_paths[0].parents[2] / "visualization" / task
        episodes = video_episodes(video_root, seed_start)

    completed = int(result["total_num"]) if result else 0
    successes = int(result["succ_num"]) if result else 0
    status = "complete" if completed == requested and len(episodes) == requested else "pending"
    if completed or episodes:
        status = "partial" if status != "complete" else status
    if (task_root / ".running").exists() and status != "complete":
        status = "running"
    if (task_root / ".failed").exists() and status != "complete":
        status = "failed"
    return {
        "task": task,
        "status": status,
        "source": str(task_root),
        "episodes_requested": requested,
        "episodes_completed": completed,
        "successes": successes,
        "failures": completed - successes,
        "success_rate": successes / completed if completed else None,
        "wilson_95": wilson(successes, completed),
        "episodes": episodes,
    }


def imported_adjust_record(path: Path, requested: int) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    completed = int(value["total_num"])
    successes = int(value["succ_num"])
    episodes = value["episodes"]
    return {
        "task": "adjust_bottle",
        "status": "complete" if completed == requested and len(episodes) == requested else "partial",
        "source": str(path.resolve()),
        "episodes_requested": requested,
        "episodes_completed": completed,
        "successes": successes,
        "failures": completed - successes,
        "success_rate": successes / completed if completed else None,
        "wilson_95": wilson(successes, completed),
        "episodes": episodes,
    }


def write_csvs(root: Path, tasks: list[dict]) -> None:
    with_rows = []
    for task in tasks:
        for episode in task["episodes"]:
            with_rows.append(
                [
                    "2026-07-24",
                    "Lingbot-VA",
                    "RoboTwin",
                    task["task"],
                    "FBFM",
                    episode["seed"],
                    str(bool(episode["success"])).lower(),
                    task["status"],
                    episode["video"],
                ]
            )
    import io

    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["date", "track", "benchmark", "task", "mode", "seed", "success", "status", "output"])
    writer.writerows(with_rows)
    atomic_write(root / "trials.csv", handle.getvalue())

    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "task",
            "status",
            "successes",
            "episodes_completed",
            "episodes_requested",
            "success_rate",
            "wilson_95_low",
            "wilson_95_high",
            "source",
        ]
    )
    for task in tasks:
        interval = task["wilson_95"] or [None, None]
        writer.writerow(
            [
                task["task"],
                task["status"],
                task["successes"],
                task["episodes_completed"],
                task["episodes_requested"],
                task["success_rate"],
                interval[0],
                interval[1],
                task["source"],
            ]
        )
    atomic_write(root / "task_summary.csv", handle.getvalue())


def render_markdown(aggregate: dict) -> str:
    def rate(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Lingbot-VA + FBFM RoboTwin live results",
        "",
        f"Updated: `{aggregate['updated_at']}`",
        "",
        f"Status: `{aggregate['status'].upper()}`",
        "",
        "| Complete tasks | Episodes | Success | Micro rate | Complete-task macro rate |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['tasks_complete']}/{aggregate['tasks_requested']} | "
            f"{aggregate['episodes_completed']}/{aggregate['episodes_requested']} | "
            f"{aggregate['successes']} | {rate(aggregate['micro_success_rate'])} | "
            f"{rate(aggregate['completed_task_macro_success_rate'])} |"
        ),
        "",
        "| Task | Status | Success | Rate | 95% Wilson CI |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for task in aggregate["tasks"]:
        interval = task["wilson_95"]
        interval_text = (
            f"{100 * interval[0]:.1f}%-{100 * interval[1]:.1f}%"
            if interval
            else "-"
        )
        lines.append(
            f"| `{task['task']}` | {task['status']} | "
            f"{task['successes']}/{task['episodes_completed']} | "
            f"{rate(task['success_rate'])} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            "Only complete 20-episode task rows are final estimates. Running or partial rows",
            "are operational progress and must not be used in paper comparisons.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--import-adjust", type=Path)
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    records = []
    for task in TASKS:
        if task == "adjust_bottle" and args.import_adjust and args.import_adjust.exists():
            record = imported_adjust_record(args.import_adjust, args.episodes_per_task)
        else:
            record = local_task_record(args.root, task, args.episodes_per_task)
        records.append(record)

    completed = sum(item["episodes_completed"] for item in records)
    successes = sum(item["successes"] for item in records)
    complete_tasks = sum(item["status"] == "complete" for item in records)
    status = "complete" if complete_tasks == len(TASKS) else "running"
    aggregate = {
        "benchmark": "RoboTwin",
        "mode": "FBFM",
        "status": status,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tasks_requested": len(TASKS),
        "tasks_complete": complete_tasks,
        "episodes_per_task": args.episodes_per_task,
        "episodes_requested": len(TASKS) * args.episodes_per_task,
        "episodes_completed": completed,
        "successes": successes,
        "failures": completed - successes,
        "micro_success_rate": successes / completed if completed else None,
        "completed_task_macro_success_rate": (
            sum(item["success_rate"] for item in records if item["status"] == "complete")
            / complete_tasks
            if complete_tasks
            else None
        ),
        "tasks": records,
    }
    atomic_write(args.root / "aggregate.json", json.dumps(aggregate, indent=2) + "\n")
    write_csvs(args.root, records)
    live_markdown = render_markdown(aggregate)
    atomic_write(args.root / "LIVE_STATUS.md", live_markdown)
    if args.paper_dir:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(
            args.paper_dir / "robotwin_fbfm_all_tasks.csv",
            (args.root / "trials.csv").read_text(encoding="utf-8"),
        )
        atomic_write(
            args.paper_dir / "robotwin_fbfm_task_summary.csv",
            (args.root / "task_summary.csv").read_text(encoding="utf-8"),
        )
        atomic_write(
            args.paper_dir / "robotwin_fbfm_live_status.md",
            live_markdown,
        )
    print(json.dumps({key: aggregate[key] for key in (
        "status", "tasks_complete", "tasks_requested", "episodes_completed",
        "episodes_requested", "successes", "micro_success_rate"
    )}))


if __name__ == "__main__":
    main()
