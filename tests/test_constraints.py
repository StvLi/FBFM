import json

import numpy as np
import torch

from dreamzero_fbfm.constraints import ActionNormalizer, ChunkConstraints, ConstraintMode


def test_constraint_modes_only_gate_masks():
    action_target = torch.ones(1, 4, 3)
    action_mask = torch.ones_like(action_target)
    state_target = torch.ones(1, 2, 2, 1, 1)
    state_mask = torch.ones(1, 1, 2, 1, 1)

    snapshots = {}
    for mode in ConstraintMode:
        context = ChunkConstraints(
            mode=mode,
            action_targets=action_target,
            action_mask=action_mask,
            state_targets=state_target,
            state_mask=state_mask,
        )
        snapshots[mode] = context.snapshot()

    assert torch.count_nonzero(snapshots[ConstraintMode.NONE][1]) == 0
    assert torch.count_nonzero(snapshots[ConstraintMode.NONE][3]) == 0
    assert torch.count_nonzero(snapshots[ConstraintMode.RTC][1]) == 0
    assert torch.equal(snapshots[ConstraintMode.RTC][3], action_mask)
    assert torch.equal(snapshots[ConstraintMode.FBFM][1], state_mask)
    assert torch.equal(snapshots[ConstraintMode.FBFM][3], action_mask)


def test_state_slots_are_single_assignment_and_versioned():
    context = ChunkConstraints(
        mode="FBFM",
        action_targets=torch.zeros(1, 4, 3),
        action_mask=torch.zeros(1, 4, 3),
        state_targets=torch.zeros(1, 2, 2, 1, 1),
        state_mask=torch.zeros(1, 1, 2, 1, 1),
    )
    assert context.update_state_slot(0, torch.tensor([[[[2.0]]], [[[3.0]]]]))
    assert not context.update_state_slot(0, torch.zeros(1, 2, 1, 1))
    targets, mask, _, _, version = context.snapshot()
    assert version == 1
    assert torch.equal(targets[:, :, 0], torch.tensor([[[[2.0]], [[3.0]]]]))
    assert mask[:, :, 0].item() == 1


def test_action_normalizer_matches_q99_and_pads(tmp_path):
    metadata = {
        "libero_sim": {
            "statistics": {
                "action": {"actions": {"q01": [-1.0, 0.0], "q99": [1.0, 2.0]}}
            }
        }
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    normalizer = ActionNormalizer.from_metadata(path, model_dim=4)
    actual = normalizer.normalize(np.asarray([[0.0, 2.0]], dtype=np.float32))
    torch.testing.assert_close(actual, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
