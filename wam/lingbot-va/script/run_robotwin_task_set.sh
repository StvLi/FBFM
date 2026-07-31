#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)
workspace=$(cd -- "$lingbot/../.." && pwd)
variant=${LINGBOT_VA_CONSTRAINT_VARIANT:-NONE}
run_root=${LINGBOT_VA_TASK_SET_ROOT:?LINGBOT_VA_TASK_SET_ROOT is required}
tasks_file=${LINGBOT_VA_TASKS_FILE:?LINGBOT_VA_TASKS_FILE is required}
episodes_per_task=${ROBOTWIN_EPISODES_PER_TASK:-20}
shard_count=${LINGBOT_VA_TASK_SET_SHARDS:-3}
env_root=${FBFM_ENV_ROOT:-$workspace/.venvs}
client_python=${ROBOTWIN_CLIENT_PYTHON:-$env_root/fbfm-robotwin/bin/python}
paper_dir=${FBFM_PAPER_EXPERIMENT_DIR:-$run_root/paper_exports}
variant_slug=${variant,,}
variant_slug=${variant_slug//-/_}
paper_prefix=${LINGBOT_VA_PAPER_PREFIX:-robotwin_lingbot_${variant_slug}}
experiment_date=${LINGBOT_VA_EXPERIMENT_DATE:-$(date +%F)}
port_base=${LINGBOT_VA_TASK_SET_PORT_BASE:-29356}
master_port_base=${LINGBOT_VA_TASK_SET_MASTER_PORT_BASE:-29361}

[[ -s $tasks_file ]] || { echo "task file is missing or empty: $tasks_file" >&2; exit 2; }
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
    echo "RoboTwin client Python is not on PATH: $requested_client_python" >&2
    echo "Set ROBOTWIN_CLIENT_PYTHON or run scripts/bootstrap/create_envs.sh --route lingbot." >&2
    exit 2
  }
fi
export ROBOTWIN_CLIENT_PYTHON="$client_python"
[[ $episodes_per_task =~ ^[1-9][0-9]*$ ]] || {
  echo "ROBOTWIN_EPISODES_PER_TASK must be a positive integer: $episodes_per_task" >&2
  exit 2
}
[[ $shard_count =~ ^[1-9][0-9]*$ ]] || {
  echo "LINGBOT_VA_TASK_SET_SHARDS must be a positive integer: $shard_count" >&2
  exit 2
}
[[ $port_base =~ ^[1-9][0-9]*$ && $master_port_base =~ ^[1-9][0-9]*$ ]] || {
  echo "task-set port bases must be positive integers" >&2
  exit 2
}
mapfile -t tasks < <(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "$tasks_file")

valid_result() {
  local task_root=$1 task=$2
  "$client_python" - "$task_root" "$task" "$episodes_per_task" <<'PY'
import json
import re
import sys
from pathlib import Path

root, task, requested = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
result_path = root / "client" / "stseed-10000" / "metrics" / task / "res.json"
video_root = root / "client" / "stseed-10000" / "visualization" / task
if not result_path.is_file() or not video_root.is_dir():
    raise SystemExit(1)
try:
    result = json.loads(result_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError, TypeError, ValueError):
    raise SystemExit(1)
indices = []
successes = 0
for path in video_root.glob("*.mp4"):
    match = re.match(r"^(\d+)_.*_(True|False)\.mp4$", path.name)
    if match:
        indices.append(int(match.group(1)))
        successes += match.group(2) == "True"
valid = (
    int(result.get("total_num", -1)) == requested
    and int(result.get("succ_num", -1)) == successes
    and sorted(indices) == list(range(requested))
)
raise SystemExit(0 if valid else 1)
PY
}

aggregate() {
  flock "$run_root/.aggregate.lock" \
    "$client_python" \
      "$script_dir/aggregate_robotwin_all_tasks.py" "$run_root" \
      --episodes-per-task "$episodes_per_task" \
      --tasks-file "$tasks_file" \
      --paper-dir "$paper_dir" \
      --mode "$variant" \
      --paper-prefix "$paper_prefix" \
      --experiment-date "$experiment_date"
}

if [[ ${1:-} == --worker ]]; then
  shard=${2:?missing shard index}
  mkdir -p "$run_root/tasks" "$run_root/logs"
  for ((index=shard; index<${#tasks[@]}; index+=shard_count)); do
    task=${tasks[$index]}
    task_root=$run_root/tasks/$task
    if valid_result "$task_root" "$task"; then
      aggregate >> "$run_root/logs/aggregation.log" 2>&1
      continue
    fi

    if [[ -d $task_root ]]; then
      interrupted_root=${task_root}.interrupted.$(date +%Y%m%d_%H%M%S)
      mv "$task_root" "$interrupted_root"
    fi

    mkdir -p "$task_root"
    rm -f "$task_root/.failed"
    touch "$task_root/.running"
    set +e
    env \
      ROBOTWIN_TASK_NAME="$task" \
      ROBOTWIN_TASK_CONFIG=demo_clean \
      ROBOTWIN_EVAL_SEED=0 \
      LINGBOT_VA_PORT="$((port_base + shard))" \
      LINGBOT_VA_MASTER_PORT="$((master_port_base + shard))" \
      LINGBOT_VA_RUN_OUTPUT="$task_root" \
      bash "$script_dir/run_constraint_ablation.sh" "$variant" "$episodes_per_task"
    task_status=$?
    set -e
    rm -f "$task_root/.running"
    if (( task_status != 0 )); then
      touch "$task_root/.failed"
      printf '%s task=%s status=%d\n' "$(date --iso-8601=seconds)" "$task" "$task_status" \
        >> "$run_root/logs/failures.log"
    fi
    aggregate >> "$run_root/logs/aggregation.log" 2>&1 || true
  done
  exit 0
fi

if (( shard_count < 1 || shard_count > 3 )); then
  echo "LINGBOT_VA_TASK_SET_SHARDS must be in [1, 3] on this GPU" >&2
  exit 2
fi

mkdir -p "$run_root/logs"
printf '%s\n' "$$" > "$run_root/launcher.pid"
aggregate | tee -a "$run_root/logs/aggregation.log"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for ((shard=0; shard<shard_count; shard++)); do
  setsid "$0" --worker "$shard" > "$run_root/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
trap - INT TERM
aggregate | tee -a "$run_root/logs/aggregation.log"
if ! "$client_python" -c \
  'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if value["status"] == "complete" else 1)' \
  "$run_root/aggregate.json"; then
  status=1
fi
exit "$status"
