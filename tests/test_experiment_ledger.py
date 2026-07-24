import json

from dreamzero_fbfm.experiment_ledger import TaskSpec, collect_rows, task_directory, write_tables


def write_episode(path, *, trial_id, success):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "success": success,
        "executed_steps": 42,
        "waves": 6,
        "elapsed_seconds": 12.5,
        "inference_wave_seconds": [1.0, 2.0],
        "actions_finite": True,
        "model_seed": trial_id,
        "environment_seed": 0,
        "task_description": "test task",
        "trajectory": f"trial_{trial_id:03d}.npz",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def test_incremental_tables_distinguish_complete_and_running_tasks(tmp_path):
    specs = [
        TaskSpec("libero_spatial", 0, "first task"),
        TaskSpec("libero_spatial", 1, "second task"),
    ]
    first = task_directory(tmp_path, specs[0]) / "episodes.jsonl"
    second = task_directory(tmp_path, specs[1]) / "episodes.jsonl"
    write_episode(first, trial_id=0, success=True)
    write_episode(first, trial_id=1, success=False)
    write_episode(second, trial_id=0, success=True)

    task_rows, trial_rows = collect_rows(tmp_path, specs, target_trials=2)
    assert task_rows[0]["status"] == "complete"
    assert task_rows[0]["success_rate"] == 0.5
    assert task_rows[1]["status"] == "running"
    assert len(trial_rows) == 3

    write_tables(tmp_path, specs, 2, mode="FBFM", code_commit="abc1234")
    assert (tmp_path / "task_summary.csv").is_file()
    assert (tmp_path / "trials.csv").is_file()
    status = (tmp_path / "live_status.md").read_text(encoding="utf-8")
    assert "1/2" in status
    assert "3/4" in status


def test_duplicate_trial_ids_are_rejected(tmp_path):
    spec = TaskSpec("libero_spatial", 0, "first task")
    episodes = task_directory(tmp_path, spec) / "episodes.jsonl"
    write_episode(episodes, trial_id=0, success=True)
    write_episode(episodes, trial_id=0, success=False)

    try:
        collect_rows(tmp_path, [spec], target_trials=20)
    except ValueError as error:
        assert "duplicate trial ids" in str(error)
    else:
        raise AssertionError("duplicate trials were accepted")
