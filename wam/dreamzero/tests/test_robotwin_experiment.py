import copy

import pytest

from wam.dreamzero.evaluation.robotwin.experiment import (
    CONFIGS,
    MODES,
    TASKS,
    aggregate,
    freeze_manifest,
    model_noise_seed,
    validate_results,
)


def _candidates(per_cell=20):
    records = []
    for task in TASKS:
        for config in CONFIGS:
            for index in range(per_cell):
                records.append(
                    {
                        "accepted": True,
                        "task": task,
                        "config": config,
                        "seed": 10_000 + index,
                        "instruction": f"instruction {index}",
                        "instruction_index": index,
                        "randomization": {"index": index} if config == "demo_randomized" else {},
                        "background_texture": "textures/frozen.png" if config == "demo_randomized" else None,
                        "background_texture_sha256": "a" * 64 if config == "demo_randomized" else None,
                    }
                )
    return records


def test_manifest_freezes_all_cells_and_chunk_noise_is_deterministic():
    manifest = freeze_manifest(_candidates())
    assert len(manifest) == 17 * 2 * 20
    assert model_noise_seed(manifest[0], 3) == model_noise_seed(manifest[0], 3)
    assert model_noise_seed(manifest[0], 3) != model_noise_seed(manifest[0], 4)


def test_manifest_can_freeze_one_task_config_for_gated_ab():
    manifest = freeze_manifest(
        _candidates(),
        tasks=["adjust_bottle"],
        configs=["demo_clean"],
    )
    assert len(manifest) == 20
    assert {(item["task"], item["config"]) for item in manifest} == {
        ("adjust_bottle", "demo_clean")
    }


def test_randomized_candidate_requires_background_checksum():
    candidates = _candidates()
    record = next(item for item in candidates if item["config"] == "demo_randomized")
    record["background_texture_sha256"] = None
    with pytest.raises(ValueError, match="texture"):
        freeze_manifest(candidates)


def test_results_reject_mode_specific_checkpoints():
    manifest = freeze_manifest(_candidates())
    episode_id = manifest[0]["episode_id"]
    results = [
        {"mode": "None", "episode_id": episode_id, "success": True, "checkpoint_sha256": "a" * 64},
        {"mode": "RTC", "episode_id": episode_id, "success": True, "checkpoint_sha256": "b" * 64},
    ]
    with pytest.raises(ValueError, match="same checkpoint"):
        validate_results(manifest, results, allow_partial=True)


def test_aggregate_uses_required_column_order_and_boolean_success():
    manifest = freeze_manifest(_candidates())
    episode_id = manifest[0]["episode_id"]
    results = [
        {"mode": mode, "episode_id": episode_id, "success": mode != "RTC", "checkpoint_sha256": "a" * 64}
        for mode in MODES
    ]
    markdown, summary = aggregate(manifest, results, allow_partial=True)
    assert markdown.splitlines()[0].startswith(
        "| Task | Baseline Clean | RTC Clean | Ours Clean | Baseline Random | RTC Random | Ours Random |"
    )
    assert summary["None"]["Clean"]["success"] == 1
    bad = copy.deepcopy(results)
    bad[0]["success"] = 1
    with pytest.raises(ValueError, match="JSON boolean"):
        validate_results(manifest, bad, allow_partial=True)
