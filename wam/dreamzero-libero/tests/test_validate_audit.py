import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_audit.py"


def _solver_step(offset: int, *, async_step: bool) -> dict:
    return {
        "event": "solver_step",
        "guided": async_step,
        "state_mask_nonzero": int(async_step),
        "action_mask_nonzero": 56 if async_step else 0,
        "context_version": offset if async_step else 0,
        "feedback_action_offsets": [offset] if async_step else [],
        "feedback_state_slots": [0] if async_step else [],
        "action_correction_norm": 1.0 if async_step else 0.0,
        "gpu_allocated_bytes": 10,
        "gpu_peak_allocated_bytes": 20,
    }


def test_validator_keeps_episode_chunks_aligned_after_partial_terminal_wave(tmp_path):
    records = []
    for async_step, evaluations in ((False, 8), (True, 7), (False, 8), (True, 8)):
        records.append({"event": "chunk_begin", "pseudo_async": async_step})
        records.extend(
            _solver_step(offset, async_step=async_step)
            for offset in range(1, evaluations + 1)
        )

    audit = tmp_path / "solver.jsonl"
    audit.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(audit),
            "--mode",
            "FBFM",
            "--minimum-async-chunks",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["async_chunks"] == 2
    assert summary["complete_async_chunks"] == 1
    assert summary["solver_steps"] == 31
