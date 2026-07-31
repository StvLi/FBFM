#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)
workspace=$(cd -- "$lingbot/../.." && pwd)
run_root=${LINGBOT_VA_EXPERIMENT_LIST_ROOT:-$lingbot/robotwin_outputs/fbfm_completion_7cells_27ep_20260729}
experiment_list=${LINGBOT_VA_EXPERIMENT_LIST:-$lingbot/config/robotwin_fbfm_completion_7cells_27ep.tsv}
paper_dir=${FBFM_PAPER_EXPERIMENT_DIR:-$run_root/paper_exports}
paper_prefix=${LINGBOT_VA_PAPER_PREFIX:-robotwin_lingbot_fbfm_completion_7cells_27ep}
max_attempts=${LINGBOT_VA_MAX_JOB_ATTEMPTS:-3}
stagger_seconds=${LINGBOT_VA_CHANNEL_STAGGER_SECONDS:-20}
port_base=${LINGBOT_VA_EXPERIMENT_PORT_BASE:-29556}
master_port_base=${LINGBOT_VA_EXPERIMENT_MASTER_PORT_BASE:-29561}
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
    echo "RoboTwin client Python is not on PATH: $requested_client_python" >&2
    echo "Set ROBOTWIN_CLIENT_PYTHON or run scripts/bootstrap/create_envs.sh --route lingbot." >&2
    exit 2
  }
fi
export ROBOTWIN_CLIENT_PYTHON="$client_python"

aggregate() {
  flock "$run_root/.aggregate.lock" \
    "$client_python" "$script_dir/aggregate_robotwin_experiment_list.py" \
      "$run_root" --experiment-list "$experiment_list" \
      --paper-dir "$paper_dir" --paper-prefix "$paper_prefix"
}

valid_result() {
  local job_root=$1 task=$2 start_seed=$3 test_num=$4
  "$client_python" - "$job_root" "$task" "$start_seed" "$test_num" <<'PY'
import json
import re
import sys
from pathlib import Path

root, task, start_seed, requested = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
result_path = root / "client" / f"stseed-{start_seed}" / "metrics" / task / "res.json"
video_root = root / "client" / f"stseed-{start_seed}" / "visualization" / task
if not result_path.is_file() or not video_root.is_dir():
    raise SystemExit(1)
result = json.loads(result_path.read_text(encoding="utf-8"))
videos = []
successes = 0
for path in video_root.glob("*.mp4"):
    match = re.match(r"^(\d+)_.*_(True|False)\.mp4$", path.name)
    if match:
        videos.append(int(match.group(1)))
        successes += match.group(2) == "True"
valid = (
    int(result.get("total_num", -1)) == requested
    and int(result.get("succ_num", -1)) == successes
    and sorted(videos) == list(range(requested))
)
raise SystemExit(0 if valid else 1)
PY
}

if [[ ${1:-} == --worker ]]; then
  channel=${2:?missing channel}
  mkdir -p "$run_root/jobs" "$run_root/logs"
  while IFS=$'\t' read -r job_id condition config task start_seed test_num assigned_channel; do
    [[ $job_id == job_id ]] && continue
    [[ $assigned_channel == "$channel" ]] || continue
    eval_seed=$((start_seed / 10000 - 1))
    job_root=$run_root/jobs/$job_id

    if valid_result "$job_root" "$task" "$start_seed" "$test_num"; then
      rm -f "$job_root/.running" "$job_root/.failed"
      touch "$job_root/.complete"
      aggregate >>"$run_root/logs/aggregation.log" 2>&1
      continue
    fi

    attempt=1
    while (( attempt <= max_attempts )); do
      if [[ -d $job_root ]]; then
        interrupted=${job_root}.interrupted.$(date +%Y%m%d_%H%M%S).attempt${attempt}
        mv "$job_root" "$interrupted"
      fi
      mkdir -p "$job_root"
      touch "$job_root/.running"
      printf '%s channel=%s job=%s attempt=%s status=start\n' \
        "$(date --iso-8601=seconds)" "$channel" "$job_id" "$attempt" \
        >>"$run_root/logs/events.log"

      set +e
      env \
        ROBOTWIN_TASK_NAME="$task" \
        ROBOTWIN_TASK_CONFIG="$config" \
        ROBOTWIN_EVAL_SEED="$eval_seed" \
        LINGBOT_VA_PORT="$((port_base + channel))" \
        LINGBOT_VA_MASTER_PORT="$((master_port_base + channel))" \
        LINGBOT_VA_RUN_OUTPUT="$job_root" \
        bash "$script_dir/run_constraint_ablation.sh" FBFM "$test_num"
      command_status=$?
      set -e
      rm -f "$job_root/.running"

      if (( command_status == 0 )) && valid_result "$job_root" "$task" "$start_seed" "$test_num"; then
        touch "$job_root/.complete"
        printf '%s channel=%s job=%s attempt=%s status=complete\n' \
          "$(date --iso-8601=seconds)" "$channel" "$job_id" "$attempt" \
          >>"$run_root/logs/events.log"
        aggregate >>"$run_root/logs/aggregation.log" 2>&1
        break
      fi

      touch "$job_root/.failed"
      printf '%s channel=%s job=%s attempt=%s status=failed exit=%s\n' \
        "$(date --iso-8601=seconds)" "$channel" "$job_id" "$attempt" "$command_status" \
        >>"$run_root/logs/events.log"
      aggregate >>"$run_root/logs/aggregation.log" 2>&1 || true
      attempt=$((attempt + 1))
    done
  done <"$experiment_list"
  exit 0
fi

[[ -s $experiment_list ]] || { echo "missing experiment list: $experiment_list" >&2; exit 2; }
[[ $max_attempts =~ ^[1-9][0-9]*$ ]] || { echo "invalid max attempts: $max_attempts" >&2; exit 2; }
[[ $stagger_seconds =~ ^[0-9]+$ ]] || { echo "invalid channel stagger: $stagger_seconds" >&2; exit 2; }
[[ $port_base =~ ^[1-9][0-9]*$ && $master_port_base =~ ^[1-9][0-9]*$ ]] \
  || { echo "experiment-list port bases must be positive integers" >&2; exit 2; }
mkdir -p "$run_root/jobs" "$run_root/logs" "$paper_dir"
cp "$experiment_list" "$run_root/experiment_list.tsv"
printf '%s\n' "$$" >"$run_root/launcher.pid"
aggregate | tee -a "$run_root/logs/aggregation.log"

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for channel in 0 1 2; do
  setsid "$0" --worker "$channel" >"$run_root/logs/channel${channel}.log" 2>&1 &
  pids+=("$!")
  if (( channel < 2 && stagger_seconds > 0 )); then
    sleep "$stagger_seconds"
  fi
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
trap - INT TERM
aggregate | tee -a "$run_root/logs/aggregation.log"

if ! "$client_python" - "$run_root/aggregate.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value["status"] == "complete" else 1)
PY
then
  status=1
fi
exit "$status"
