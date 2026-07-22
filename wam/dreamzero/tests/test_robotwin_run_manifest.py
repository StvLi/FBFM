import pathlib
import sys

import pytest


DREAMZERO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DREAMZERO_ROOT))

from evaluation.robotwin.run_manifest import select_shard  # noqa: E402


def test_select_shard_is_disjoint_complete_and_ordered():
    episodes = [{"episode_id": str(index)} for index in range(20)]
    left = select_shard(episodes, shard_index=0, num_shards=2)
    right = select_shard(episodes, shard_index=1, num_shards=2)
    assert [item["episode_id"] for item in left] == [str(index) for index in range(0, 20, 2)]
    assert [item["episode_id"] for item in right] == [str(index) for index in range(1, 20, 2)]
    assert {item["episode_id"] for item in left}.isdisjoint(
        item["episode_id"] for item in right
    )
    assert len(left) + len(right) == len(episodes)


@pytest.mark.parametrize("shard_index,num_shards", [(-1, 2), (2, 2), (0, 0)])
def test_select_shard_rejects_invalid_coordinates(shard_index, num_shards):
    with pytest.raises(ValueError):
        select_shard([], shard_index=shard_index, num_shards=num_shards)
