"""Incremental result tables for the full LIBERO benchmark."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int
    description: str


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def task_directory(output: Path, spec: TaskSpec) -> Path:
    return output / "tasks" / spec.suite / f"task_{spec.task_id:03d}"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return center - margin, center + margin


def collect_rows(
    output: Path, specs: Iterable[TaskSpec], target_trials: int
) -> tuple[list[dict], list[dict]]:
    task_rows: list[dict] = []
    trial_rows: list[dict] = []
    for spec in specs:
        directory = task_directory(output, spec)
        records = load_jsonl(directory / "episodes.jsonl")
        trial_ids = [int(record["trial_id"]) for record in records]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError(f"duplicate trial ids in {directory}")
        if any(trial_id < 0 or trial_id >= target_trials for trial_id in trial_ids):
            raise ValueError(f"out-of-range trial id in {directory}: {trial_ids}")
        successes = sum(bool(record["success"]) for record in records)
        trials = len(records)
        low, high = wilson_interval(successes, trials)
        status = "complete" if trials == target_trials else ("running" if trials else "pending")
        wave_seconds = [
            float(value)
            for record in records
            for value in record.get("inference_wave_seconds", [])
        ]
        task_rows.append(
            {
                "suite": spec.suite,
                "task_id": spec.task_id,
                "status": status,
                "trials": trials,
                "target_trials": target_trials,
                "successes": successes,
                "failures": trials - successes,
                "success_rate": successes / trials if trials else None,
                "wilson_low": low,
                "wilson_high": high,
                "mean_executed_steps": (
                    sum(int(record["executed_steps"]) for record in records) / trials
                    if trials
                    else None
                ),
                "mean_elapsed_seconds": (
                    sum(float(record["elapsed_seconds"]) for record in records) / trials
                    if trials
                    else None
                ),
                "mean_inference_wave_seconds": (
                    sum(wave_seconds) / len(wave_seconds) if wave_seconds else None
                ),
                "task_description": spec.description,
                "output": str(directory.relative_to(output)),
            }
        )
        for record in records:
            waves = [float(value) for value in record.get("inference_wave_seconds", [])]
            trial_rows.append(
                {
                    "suite": spec.suite,
                    "task_id": spec.task_id,
                    "trial_id": int(record["trial_id"]),
                    "success": bool(record["success"]),
                    "executed_steps": int(record["executed_steps"]),
                    "waves": int(record["waves"]),
                    "elapsed_seconds": float(record["elapsed_seconds"]),
                    "mean_inference_wave_seconds": sum(waves) / len(waves) if waves else None,
                    "actions_finite": bool(record["actions_finite"]),
                    "model_seed": int(record["model_seed"]),
                    "environment_seed": int(record["environment_seed"]),
                    "task_description": record["task_description"],
                    "trajectory": record["trajectory"],
                }
            )
    return task_rows, trial_rows


def _format_rate(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table to {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_tables(
    output: Path,
    specs: Iterable[TaskSpec],
    target_trials: int,
    *,
    mode: str,
    code_commit: str,
) -> tuple[list[dict], list[dict]]:
    output.mkdir(parents=True, exist_ok=True)
    task_rows, trial_rows = collect_rows(output, specs, target_trials)
    _write_csv(output / "task_summary.csv", task_rows)
    if trial_rows:
        _write_csv(output / "trials.csv", trial_rows)

    suites = list(dict.fromkeys(row["suite"] for row in task_rows))
    total_trials = sum(row["trials"] for row in task_rows)
    total_successes = sum(row["successes"] for row in task_rows)
    complete_tasks = sum(row["status"] == "complete" for row in task_rows)
    lines = [
        "# DreamZero + FBFM LIBERO live results",
        "",
        f"Updated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        f"Mode: `{mode}` | Code: `{code_commit}` | Target: {len(task_rows)} tasks x {target_trials} trials",
        "",
        "| Complete tasks | Episodes | Success | Micro rate |",
        "| ---: | ---: | ---: | ---: |",
        f"| {complete_tasks}/{len(task_rows)} | {total_trials}/{len(task_rows) * target_trials} | {total_successes} | {_format_rate(total_successes / total_trials if total_trials else None)} |",
        "",
        "## Suite progress",
        "",
        "| Suite | Complete tasks | Episodes | Success | Rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for suite in suites:
        rows = [row for row in task_rows if row["suite"] == suite]
        trials = sum(row["trials"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        complete = sum(row["status"] == "complete" for row in rows)
        lines.append(
            f"| `{suite}` | {complete}/{len(rows)} | {trials}/{len(rows) * target_trials} | {successes} | {_format_rate(successes / trials if trials else None)} |"
        )
    lines.extend(
        [
            "",
            "## Task results",
            "",
            "| Suite | Task | Status | Success | Rate | 95% Wilson CI | Mean steps | Mean seconds | Description |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in task_rows:
        interval = (
            "-"
            if row["wilson_low"] is None
            else f"{100 * row['wilson_low']:.1f}%-{100 * row['wilson_high']:.1f}%"
        )
        mean_steps = "-" if row["mean_executed_steps"] is None else f"{row['mean_executed_steps']:.1f}"
        mean_seconds = "-" if row["mean_elapsed_seconds"] is None else f"{row['mean_elapsed_seconds']:.2f}"
        lines.append(
            f"| `{row['suite']}` | {row['task_id']} | {row['status']} | {row['successes']}/{row['trials']} | {_format_rate(row['success_rate'])} | {interval} | {mean_steps} | {mean_seconds} | {row['task_description']} |"
        )
    lines.extend(
        [
            "",
            "Only rows with status `complete` are final 20-trial task estimates.",
            "Running rows are progress indicators and must not be used as paper results.",
            "",
        ]
    )
    temporary = output / "live_status.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "live_status.md")
    return task_rows, trial_rows
