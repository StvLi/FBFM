#!/usr/bin/env python3
"""Aggregate a heterogeneous list of RoboTwin continuation experiments."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path


VIDEO_RE = re.compile(r"^(\d+)_.*_(True|False)\.mp4$")


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0 or successes < 0 or successes > total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - half_width, center + half_width]


def load_jobs(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "job_id",
        "condition",
        "config",
        "task",
        "start_seed",
        "test_num",
        "channel",
    }
    if not jobs or set(jobs[0]) != required:
        raise ValueError(f"unexpected experiment-list columns in {path}")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("experiment list contains duplicate job_id values")
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job["job_id"]):
            raise ValueError(f"unsafe job_id: {job['job_id']!r}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", job["task"]):
            raise ValueError(f"unsafe task name: {job['task']!r}")
        job["start_seed"] = int(job["start_seed"])
        job["test_num"] = int(job["test_num"])
        job["channel"] = int(job["channel"])
        if job["condition"] not in {"clean", "random"}:
            raise ValueError(f"unknown condition: {job['condition']}")
        expected_config = {
            "clean": "demo_clean",
            "random": "demo_randomized",
        }[job["condition"]]
        if job["config"] != expected_config:
            raise ValueError(f"condition/config mismatch: {job}")
        if job["start_seed"] % 10000 or job["start_seed"] < 10000:
            raise ValueError(f"invalid RoboTwin start seed: {job['start_seed']}")
        if job["test_num"] <= 0 or job["channel"] not in {0, 1, 2}:
            raise ValueError(f"invalid job coordinates: {job}")
    return jobs


def read_videos(root: Path, start_seed: int) -> tuple[list[dict], list[int]]:
    episodes = []
    indices = []
    if not root.is_dir():
        return episodes, indices
    for path in root.glob("*.mp4"):
        match = VIDEO_RE.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        indices.append(index)
        episodes.append(
            {
                "episode_index": index,
                "seed": start_seed + index,
                "success": match.group(2) == "True",
                "video": str(path.resolve()),
            }
        )
    return sorted(episodes, key=lambda item: item["episode_index"]), sorted(indices)


def job_record(root: Path, job: dict) -> dict:
    job_root = root / "jobs" / job["job_id"]
    result_path = (
        job_root
        / "client"
        / f"stseed-{job['start_seed']}"
        / "metrics"
        / job["task"]
        / "res.json"
    )
    video_root = (
        job_root
        / "client"
        / f"stseed-{job['start_seed']}"
        / "visualization"
        / job["task"]
    )
    result = None
    result_error = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            result_error = str(error)
    try:
        completed = int(result.get("total_num", 0)) if result else 0
        successes = int(result.get("succ_num", 0)) if result else 0
    except (TypeError, ValueError) as error:
        result_error = str(error)
        completed = 0
        successes = 0
    episodes, indices = read_videos(video_root, job["start_seed"])
    expected_indices = list(range(job["test_num"]))
    count_consistent = (
        completed == job["test_num"]
        and successes == sum(int(item["success"]) for item in episodes)
        and indices == expected_indices
    )
    status = "complete" if count_consistent else "pending"
    if completed or episodes or result_error:
        status = "partial" if not count_consistent else status
    if (job_root / ".running").exists() and status != "complete":
        status = "running"
    if (job_root / ".failed").exists() and status != "complete":
        status = "failed"
    interval = wilson(successes, completed)
    return {
        **job,
        "status": status,
        "job_root": str(job_root.resolve()),
        "result_path": str(result_path.resolve()),
        "result_error": result_error,
        "episodes_requested": job["test_num"],
        "episodes_completed": completed,
        "successes": successes,
        "failures": completed - successes,
        "success_rate": successes / completed if completed else None,
        "wilson_95": interval,
        "video_count": len(episodes),
        "expected_seeds": list(
            range(job["start_seed"], job["start_seed"] + job["test_num"])
        ),
        "episodes": episodes,
        "count_consistent": count_consistent,
    }


def write_csvs(root: Path, records: list[dict]) -> None:
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "condition",
            "config",
            "task",
            "start_seed",
            "test_num",
            "channel",
            "status",
            "successes",
            "episodes_completed",
            "success_rate",
            "wilson_95_low",
            "wilson_95_high",
            "result_path",
        ]
    )
    for record in records:
        interval = record["wilson_95"] or [None, None]
        writer.writerow(
            [
                record["condition"],
                record["config"],
                record["task"],
                record["start_seed"],
                record["test_num"],
                record["channel"],
                record["status"],
                record["successes"],
                record["episodes_completed"],
                record["success_rate"],
                interval[0],
                interval[1],
                record["result_path"],
            ]
        )
    atomic_write(root / "job_summary.csv", handle.getvalue())

    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "method",
            "condition",
            "config",
            "task",
            "seed",
            "success",
            "job_id",
            "channel",
            "video",
        ]
    )
    for record in records:
        for episode in record["episodes"]:
            writer.writerow(
                [
                    "FBFM",
                    record["condition"],
                    record["config"],
                    record["task"],
                    episode["seed"],
                    str(episode["success"]).lower(),
                    record["job_id"],
                    record["channel"],
                    episode["video"],
                ]
            )
    atomic_write(root / "trials.csv", handle.getvalue())


def markdown(aggregate: dict) -> str:
    def rate(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# LingBot-VA FBFM continuation experiments",
        "",
        f"Updated: `{aggregate['updated_at']}`",
        "",
        f"Status: `{aggregate['status'].upper()}`",
        "",
        "| Jobs | Episodes | Success | Micro rate | Running | Failed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['jobs_complete']}/{aggregate['jobs_requested']} | "
            f"{aggregate['episodes_completed']}/{aggregate['episodes_requested']} | "
            f"{aggregate['successes']} | {rate(aggregate['micro_success_rate'])} | "
            f"{aggregate['jobs_running']} | {aggregate['jobs_failed']} |"
        ),
        "",
        "| Condition | Task | Seed range | Channel | Status | Success | Rate |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for record in aggregate["jobs"]:
        end_seed = record["start_seed"] + record["test_num"] - 1
        lines.append(
            f"| `{record['condition']}` | `{record['task']}` | "
            f"{record['start_seed']}-{end_seed} | {record['channel']} | "
            f"{record['status']} | {record['successes']}/{record['episodes_completed']} | "
            f"{rate(record['success_rate'])} |"
        )
    lines.extend(
        [
            "",
            "A row is complete only when res.json totals, per-episode videos, indices,",
            "success labels, and the requested seed range all agree.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--experiment-list", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument(
        "--paper-prefix", default="robotwin_lingbot_fbfm_completion_7cells_27ep"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.paper_prefix):
        raise ValueError(f"paper prefix must be a safe file-name component: {args.paper_prefix!r}")

    records = [job_record(args.root, job) for job in load_jobs(args.experiment_list)]
    episodes_completed = sum(record["episodes_completed"] for record in records)
    successes = sum(record["successes"] for record in records)
    jobs_complete = sum(record["status"] == "complete" for record in records)
    aggregate = {
        "benchmark": "RoboTwin",
        "model": "LingBot-VA",
        "method": "FBFM",
        "status": "complete" if jobs_complete == len(records) else "running",
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs_requested": len(records),
        "jobs_complete": jobs_complete,
        "jobs_running": sum(record["status"] == "running" for record in records),
        "jobs_failed": sum(record["status"] == "failed" for record in records),
        "episodes_requested": sum(record["test_num"] for record in records),
        "episodes_completed": episodes_completed,
        "successes": successes,
        "failures": episodes_completed - successes,
        "micro_success_rate": successes / episodes_completed if episodes_completed else None,
        "jobs": records,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    atomic_write(args.root / "aggregate.json", json.dumps(aggregate, indent=2) + "\n")
    write_csvs(args.root, records)
    live_status = markdown(aggregate)
    atomic_write(args.root / "LIVE_STATUS.md", live_status)
    if args.paper_dir:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        for source, suffix in (
            (args.root / "job_summary.csv", "job_summary.csv"),
            (args.root / "trials.csv", "trials.csv"),
            (args.root / "LIVE_STATUS.md", "live_status.md"),
            (args.experiment_list, "experiment_list.tsv"),
            (args.root / "manifest.json", "manifest.json"),
        ):
            if not source.is_file():
                continue
            atomic_write(
                args.paper_dir / f"{args.paper_prefix}_{suffix}",
                source.read_text(encoding="utf-8"),
            )
    print(
        json.dumps(
            {
                key: aggregate[key]
                for key in (
                    "status",
                    "jobs_complete",
                    "jobs_requested",
                    "episodes_completed",
                    "episodes_requested",
                    "successes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
