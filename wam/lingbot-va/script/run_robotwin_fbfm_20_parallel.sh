#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)
run_root=${LINGBOT_VA_FBFM20_ROOT:-$lingbot/robotwin_outputs/fbfm_20_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$run_root"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch_shard() {
  local shard=$1
  local eval_seed=$2
  local count=$3
  local port=$4
  local master_port=$5
  local shard_root=$run_root/shard${shard}
  mkdir -p "$shard_root"
  setsid env \
    ROBOTWIN_EVAL_SEED="$eval_seed" \
    LINGBOT_VA_PORT="$port" \
    LINGBOT_VA_MASTER_PORT="$master_port" \
    LINGBOT_VA_ABLATION_ROOT="$shard_root" \
    PYTHONFAULTHANDLER=1 \
    bash "$script_dir/run_robotwin_fbfm.sh" "$count" \
    >"$shard_root/launcher.log" 2>&1 &
  pids+=("$!")
}

# Each shard is a fully independent model, simulator, KV cache, and feedback
# stream. The fixed pseudo-clock is identical; only the episode seed block differs.
launch_shard 0 0 7 29256 29261
sleep 2
launch_shard 1 1 7 29257 29262
sleep 2
launch_shard 2 2 6 29258 29263

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
trap - INT TERM
(( status == 0 )) || exit 1

/home/oem/tmp_ws/conda-envs/fbfm-robotwin/bin/python - "$run_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed_blocks = {"shard0": 10000, "shard1": 20000, "shard2": 30000}
aggregate = {
    "task": "adjust_bottle",
    "mode": "FBFM",
    "shards": [],
    "episodes": [],
    "succ_num": 0,
    "total_num": 0,
}

for shard in sorted(root.glob("shard*")):
    result_paths = list(shard.rglob("res.json"))
    if len(result_paths) != 1:
        raise RuntimeError(f"expected one res.json in {shard}, got {result_paths}")
    result_path = result_paths[0]
    result = json.loads(result_path.read_text())
    videos = sorted(result_path.parents[2].rglob("*.mp4"))
    seed_start = seed_blocks[shard.name]
    shard_record = {
        "name": shard.name,
        "seed_start": seed_start,
        "result": str(result_path),
        **result,
    }
    aggregate["shards"].append(shard_record)
    aggregate["succ_num"] += int(result["succ_num"])
    aggregate["total_num"] += int(result["total_num"])
    for index, video in enumerate(videos):
        aggregate["episodes"].append(
            {
                "seed": seed_start + index,
                "success": video.stem.endswith("_True"),
                "video": str(video),
            }
        )

if aggregate["total_num"] != 20 or len(aggregate["episodes"]) != 20:
    raise RuntimeError(f"incomplete FBFM-20 result: {aggregate}")
aggregate["succ_rate"] = aggregate["succ_num"] / aggregate["total_num"]
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
print(root)
PY
