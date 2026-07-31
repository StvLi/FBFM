#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)

export LINGBOT_VA_CONSTRAINT_VARIANT=${LINGBOT_VA_CONSTRAINT_VARIANT:-NONE}
export LINGBOT_VA_TASK_SET_ROOT=${LINGBOT_VA_TASK_SET_ROOT:-$lingbot/robotwin_outputs/base_none_12x20_$(date +%Y%m%d_%H%M%S)}
export LINGBOT_VA_TASKS_FILE=${LINGBOT_VA_TASKS_FILE:-$lingbot/config/robotwin_base_none_tasks_12.txt}
export ROBOTWIN_EPISODES_PER_TASK=${ROBOTWIN_EPISODES_PER_TASK:-20}
export LINGBOT_VA_TASK_SET_SHARDS=${LINGBOT_VA_TASK_SET_SHARDS:-3}
export LINGBOT_VA_TASK_SET_PORT_BASE=${LINGBOT_VA_TASK_SET_PORT_BASE:-29356}
export LINGBOT_VA_TASK_SET_MASTER_PORT_BASE=${LINGBOT_VA_TASK_SET_MASTER_PORT_BASE:-29361}
export FBFM_PAPER_EXPERIMENT_DIR=${FBFM_PAPER_EXPERIMENT_DIR:-$LINGBOT_VA_TASK_SET_ROOT/paper_exports}
export LINGBOT_VA_PAPER_PREFIX=${LINGBOT_VA_PAPER_PREFIX:-robotwin_lingbot_base_none_12x20}
export LINGBOT_VA_EXPERIMENT_DATE=${LINGBOT_VA_EXPERIMENT_DATE:-$(date +%F)}

exec "$lingbot/script/run_robotwin_task_set.sh" "$@"
