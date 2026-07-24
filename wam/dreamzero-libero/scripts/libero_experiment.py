#!/usr/bin/env python3
"""Run matched pseudo-asynchronous DreamZero FBFM episodes in LIBERO."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from dreamzero_fbfm.client import FBFMClient
from dreamzero_fbfm.pseudo_clock import solver_grants


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-workspace", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--mode", choices=("NONE", "RTC", "FBFM"), required=True)
    parser.add_argument("--state-weight", type=float, default=56 / 9600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials <= 0 or args.max_steps <= 0:
        raise ValueError("trials and max-steps must be positive")

    sys.path.insert(0, str(args.base_workspace.resolve()))
    from fbfm.libero_observation import LIBERO_DUMMY_ACTION, extract_libero_observation
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.trial_start + args.trials > len(initial_states):
        raise ValueError("requested trials exceed available official init states")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # DreamZero keeps 16 native UniPC scheduler steps but evaluates the DiT only
    # eight times according to its released cache mask.
    grants = solver_grants(simulation_steps=8, solver_steps=8)
    client = FBFMClient(args.host, args.port)
    episode_path = args.output / "episodes.jsonl"
    existing_records = load_jsonl(episode_path)
    existing_trials = {int(record["trial_id"]) for record in existing_records}
    requested_trials = set(range(args.trial_start, args.trial_start + args.trials))
    overlap = sorted(existing_trials & requested_trials)
    if overlap:
        raise ValueError(f"output already contains requested trial ids: {overlap}")
    records: list[dict] = []
    try:
        for trial_id in range(args.trial_start, args.trial_start + args.trials):
            client.reset(task.language, args.seed + trial_id)
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256
            )
            started = time.perf_counter()
            executed: list[np.ndarray] = []
            inference_seconds: list[float] = []
            success = False
            waves = 0
            try:
                env.seed(args.seed)
                env.reset()
                observation = env.set_init_state(initial_states[trial_id])
                for _ in range(args.settle_steps):
                    observation, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
                model_observation = extract_libero_observation(observation)
                inference_started = time.perf_counter()
                initial_chunk = client.predict_sync(
                    model_observation["main_image"],
                    model_observation["wrist_image"],
                    model_observation["state"],
                )
                inference_seconds.append(time.perf_counter() - inference_started)
                execution_actions = initial_chunk[:8]

                while len(executed) < args.max_steps and not success:
                    anchor = extract_libero_observation(observation)
                    client.start_predict(
                        anchor["main_image"], anchor["wrist_image"], anchor["state"], execution_actions
                    )
                    wave_started = time.perf_counter()
                    interrupted = False
                    for offset, (action, grant_count) in enumerate(
                        zip(execution_actions, grants), start=1
                    ):
                        observation, reward, done, _ = env.step(action.tolist())
                        executed.append(np.asarray(action, dtype=np.float32))
                        feedback = extract_libero_observation(observation)
                        client.feedback(
                            offset,
                            feedback["main_image"],
                            feedback["wrist_image"],
                            feedback["state"],
                        )
                        client.grant(grant_count)
                        success = bool(done or reward > 0)
                        if success or len(executed) >= args.max_steps:
                            client.cancel()
                            interrupted = True
                            break
                    inference_seconds.append(time.perf_counter() - wave_started)
                    waves += 1
                    if interrupted:
                        break
                    next_chunk = client.result()
                    execution_actions = next_chunk[8:16]
            finally:
                env.close()

            record = {
                "status": "ok",
                "mode": args.mode,
                "suite": args.suite,
                "task_id": args.task_id,
                "trial_id": trial_id,
                "environment_seed": args.seed,
                "model_seed": args.seed + trial_id,
                "task_description": task.language,
                "success": success,
                "executed_steps": len(executed),
                "waves": waves,
                "elapsed_seconds": time.perf_counter() - started,
                "inference_wave_seconds": inference_seconds,
                "actions_finite": bool(np.isfinite(np.asarray(executed)).all()),
                "protocol": {
                    "H": 16,
                    "d": 8,
                    "s": 8,
                    "scheduler_steps": 16,
                    "dit_evaluations": 8,
                    "state_weight": args.state_weight,
                    "solver_grants": list(grants),
                },
            }
            trajectory_path = args.output / "trajectories" / f"trial_{trial_id:03d}.npz"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trajectory_path,
                actions=np.asarray(executed, dtype=np.float32).reshape(-1, 7),
                success=np.asarray(success),
                trial_id=np.asarray(trial_id),
                environment_seed=np.asarray(args.seed),
                model_seed=np.asarray(args.seed + trial_id),
                task_description=np.asarray(task.language),
            )
            record["trajectory"] = str(trajectory_path.resolve())
            append_jsonl(episode_path, record)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        client.close()

    all_records = load_jsonl(episode_path)
    successes = sum(int(record["success"]) for record in all_records)
    summary = {
        "mode": args.mode,
        "suite": args.suite,
        "task_id": args.task_id,
        "trials": len(all_records),
        "successes": successes,
        "success_rate": successes / len(all_records),
        "episodes": str(episode_path.resolve()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
