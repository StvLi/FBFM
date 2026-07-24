#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)
run_root=${LINGBOT_VA_ALL_TASKS_ROOT:-$lingbot/robotwin_outputs/fbfm_all_tasks_20}
episodes_per_task=${ROBOTWIN_EPISODES_PER_TASK:-20}
shard_count=${LINGBOT_VA_ALL_TASK_SHARDS:-3}
import_adjust=${LINGBOT_VA_ADJUST_BOTTLE_AGGREGATE:-$lingbot/robotwin_outputs/fbfm_20_20260724_102818/aggregate.json}
paper_dir=${FBFM_PAPER_EXPERIMENT_DIR:-/home/oem/tmp_ws/aaai_paper/experiments}

tasks=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic
  hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away
  move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
  place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
  place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
  place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket
  place_object_scale place_object_stand place_phone_stand place_shoe press_stapler
  put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle
  shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three
  stack_bowls_two stamp_seal turn_switch
)

aggregate() {
  flock "$run_root/.aggregate.lock" \
    /home/oem/tmp_ws/conda-envs/fbfm-robotwin/bin/python \
      "$script_dir/aggregate_robotwin_all_tasks.py" "$run_root" \
      --episodes-per-task "$episodes_per_task" --import-adjust "$import_adjust" \
      --paper-dir "$paper_dir"
}

if [[ ${1:-} == --worker ]]; then
  shard=${2:?missing shard index}
  mkdir -p "$run_root/tasks" "$run_root/logs"
  for ((index=shard; index<${#tasks[@]}; index+=shard_count)); do
    task=${tasks[$index]}
    if [[ $task == adjust_bottle && -s $import_adjust ]]; then
      continue
    fi
    task_root=$run_root/tasks/$task
    result=$task_root/client/stseed-10000/metrics/$task/res.json
    if [[ -s $result ]] && /home/oem/tmp_ws/conda-envs/fbfm-robotwin/bin/python -c \
      'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d["total_num"]) == int(sys.argv[2]) else 1)' \
      "$result" "$episodes_per_task"; then
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
      LINGBOT_VA_PORT="$((29256 + shard))" \
      LINGBOT_VA_MASTER_PORT="$((29261 + shard))" \
      LINGBOT_VA_RUN_OUTPUT="$task_root" \
      bash "$script_dir/run_constraint_ablation.sh" FBFM "$episodes_per_task"
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
  echo "LINGBOT_VA_ALL_TASK_SHARDS must be in [1, 3] on this GPU" >&2
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
exit "$status"
