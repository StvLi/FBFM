#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)
workspace=$(cd -- "$lingbot/../.." && pwd)
env_root=${FBFM_ENV_ROOT:-$workspace/.venvs}
client_python=${ROBOTWIN_CLIENT_PYTHON:-$env_root/fbfm-robotwin/bin/python}
if [[ $client_python == */* ]]; then
  [[ -x $client_python ]] || {
    echo "RoboTwin client Python is not executable: $client_python" >&2
    echo "Set ROBOTWIN_CLIENT_PYTHON or run scripts/bootstrap/create_envs.sh --route lingbot." >&2
    exit 2
  }
  client_python=$(cd -- "$(dirname -- "$client_python")" && pwd)/$(basename -- "$client_python")
else
  requested_client_python=$client_python
  client_python=$(command -v "$requested_client_python") || {
    echo "RoboTwin client Python is not on PATH: $requested_client_python; set ROBOTWIN_CLIENT_PYTHON." >&2
    exit 2
  }
fi
export ROBOTWIN_CLIENT_PYTHON="$client_python"

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

"$client_python" - "$run_root" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed_blocks = {"shard0": 10000, "shard1": 20000, "shard2": 30000}
episode_counts = {"shard0": 7, "shard1": 7, "shard2": 6}
aggregate = {
    "task": "adjust_bottle",
    "mode": "FBFM",
    "shards": [],
    "episodes": [],
    "succ_num": 0,
    "total_num": 0,
    "state_guided_records": 0,
    "action_guided_records": 0,
    "numerical_failure_records": 0,
}

for shard in sorted(root.glob("shard*")):
    result_paths = list(shard.rglob("res.json"))
    if len(result_paths) != 1:
        raise RuntimeError(f"expected one res.json in {shard}, got {result_paths}")
    result_path = result_paths[0]
    result = json.loads(result_path.read_text())
    videos = sorted(
        result_path.parents[2].rglob("*.mp4"),
        key=lambda path: int(path.name.split("_", 1)[0]),
    )
    expected_count = episode_counts.get(shard.name)
    if expected_count is None:
        raise RuntimeError(f"unexpected shard directory: {shard}")
    indices = [int(path.name.split("_", 1)[0]) for path in videos]
    video_successes = sum(path.stem.endswith("_True") for path in videos)
    if (
        int(result.get("total_num", -1)) != expected_count
        or int(result.get("succ_num", -1)) != video_successes
        or indices != list(range(expected_count))
    ):
        raise RuntimeError(f"inconsistent result/video records in {shard}")
    server_logs = list(shard.rglob("logs/server.log"))
    if len(server_logs) != 1:
        raise RuntimeError(f"expected one server.log in {shard}, got {server_logs}")
    server_text = server_logs[0].read_text(errors="replace")
    state_guided = len(
        re.findall(r"phase=video[^\n]*guided=True[^\n]*mask_nonzero=[1-9]", server_text)
    )
    action_guided = len(
        re.findall(r"phase=action[^\n]*guided=True[^\n]*mask_nonzero=[1-9]", server_text)
    )
    numerical_failures = len(
        re.findall(r"non-finite|CUDA out of memory|Fatal Python error", server_text)
    )
    memory = [
        tuple(map(float, match))
        for match in re.findall(
            r"allocated_mib=([0-9.]+).*peak_allocated_mib=([0-9.]+)",
            server_text,
        )
    ]
    seed_start = seed_blocks[shard.name]
    shard_record = {
        "name": shard.name,
        "seed_start": seed_start,
        "result": str(result_path),
        "state_guided_records": state_guided,
        "action_guided_records": action_guided,
        "numerical_failure_records": numerical_failures,
        "post_chunk_allocated_mib_min": min(value[0] for value in memory),
        "post_chunk_allocated_mib_max": max(value[0] for value in memory),
        "peak_allocated_mib_max": max(value[1] for value in memory),
        **result,
    }
    aggregate["shards"].append(shard_record)
    aggregate["succ_num"] += int(result["succ_num"])
    aggregate["total_num"] += int(result["total_num"])
    aggregate["state_guided_records"] += state_guided
    aggregate["action_guided_records"] += action_guided
    aggregate["numerical_failure_records"] += numerical_failures
    for video in videos:
        index = int(video.name.split("_", 1)[0])
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
if sum(item["success"] for item in aggregate["episodes"]) != aggregate["succ_num"]:
    raise RuntimeError("video outcomes disagree with res.json success counts")
z = 1.959963984540054
n = aggregate["total_num"]
p = aggregate["succ_rate"]
denominator = 1 + z * z / n
center = (p + z * z / (2 * n)) / denominator
half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
aggregate["wilson_95"] = [center - half_width, center + half_width]
aggregate["post_chunk_allocated_mib_min"] = min(
    item["post_chunk_allocated_mib_min"] for item in aggregate["shards"]
)
aggregate["post_chunk_allocated_mib_max"] = max(
    item["post_chunk_allocated_mib_max"] for item in aggregate["shards"]
)
aggregate["peak_allocated_mib_max"] = max(
    item["peak_allocated_mib_max"] for item in aggregate["shards"]
)
(root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
print(root)
PY
