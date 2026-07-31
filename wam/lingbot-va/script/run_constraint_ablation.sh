#!/usr/bin/env bash
set -euo pipefail

# Usage: run_constraint_ablation.sh NONE|RTC|FBFM|FBFM-static [test_num]
variant=${1:?expected NONE, RTC, FBFM, or FBFM-static}
test_num=${2:-1}

case "$variant" in
  NONE|RTC)
    constraint_mode=$variant
    feedback_live=1
    ;;
  FBFM)
    constraint_mode=FBFM
    feedback_live=1
    ;;
  FBFM-static)
    constraint_mode=FBFM
    feedback_live=0
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

[[ $test_num =~ ^[1-9][0-9]*$ ]] || {
  echo "test_num must be a positive integer: $test_num" >&2
  exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lingbot=$(cd -- "$script_dir/.." && pwd)
workspace=$(cd -- "$lingbot/../.." && pwd)
external_root=${FBFM_EXTERNAL_ROOT:-$workspace/external}
env_root=${FBFM_ENV_ROOT:-$workspace/.venvs}
robotwin=${ROBOTWIN_ROOT:-$external_root/RoboTwin}
server_python=${LINGBOT_SERVER_PYTHON:-$env_root/fbfm-lingbot-va/bin/python}
client_python=${ROBOTWIN_CLIENT_PYTHON:-$env_root/fbfm-robotwin/bin/python}
model=${LINGBOT_VA_MODEL:-${FBFM_LINGBOT_VA_MODEL:-}}
port=${LINGBOT_VA_PORT:-29156}
master_port=${LINGBOT_VA_MASTER_PORT:-29161}
server_gpu=${LINGBOT_SERVER_GPU:-0}
client_gpu=${ROBOTWIN_CLIENT_GPU:-$server_gpu}
eval_seed=${ROBOTWIN_EVAL_SEED:-0}
task_name=${ROBOTWIN_TASK_NAME:-adjust_bottle}
task_config=${ROBOTWIN_TASK_CONFIG:-demo_clean}

[[ $eval_seed =~ ^[0-9]+$ ]] || {
  echo "ROBOTWIN_EVAL_SEED must be a non-negative integer: $eval_seed" >&2
  exit 2
}

resolve_executable() {
  local label=$1 value=$2 resolved
  if [[ $value == */* ]]; then
    [[ -x $value ]] || {
      echo "$label is not executable: $value" >&2
      return 2
    }
    resolved=$(cd -- "$(dirname -- "$value")" && pwd)/$(basename -- "$value")
  else
    resolved=$(command -v "$value") || {
      echo "$label is not on PATH: $value" >&2
      return 2
    }
  fi
  printf '%s\n' "$resolved"
}

server_python=$(resolve_executable "LingBot server Python" "$server_python") || {
  echo "Set LINGBOT_SERVER_PYTHON or run scripts/bootstrap/create_envs.sh --route lingbot." >&2
  exit 2
}
client_python=$(resolve_executable "RoboTwin client Python" "$client_python") || {
  echo "Set ROBOTWIN_CLIENT_PYTHON or run scripts/bootstrap/create_envs.sh --route lingbot." >&2
  exit 2
}
[[ -d $robotwin ]] || {
  echo "RoboTwin checkout is missing: $robotwin (set ROBOTWIN_ROOT)." >&2
  exit 2
}
robotwin=$(cd -- "$robotwin" && pwd)
[[ -n $model && -d $model ]] || {
  echo "LingBot checkpoint directory is missing; set LINGBOT_VA_MODEL." >&2
  exit 2
}
model=$(cd -- "$model" && pwd)
server_prefix=$(cd -- "$(dirname -- "$server_python")/.." && pwd)
client_site=$("$client_python" -c 'import site; print(site.getsitepackages()[0])')
vulkan_icd=${ROBOTWIN_VK_ICD_FILENAMES:-${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}}
output=${LINGBOT_VA_RUN_OUTPUT:-${LINGBOT_VA_ABLATION_ROOT:-$lingbot/robotwin_outputs}/${task_name}_${variant}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$output/logs" "$output/server_debug"
output=$(cd -- "$output" && pwd)

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

port_is_listening() {
  [[ -n $(ss -ltnH "sport = :$port" 2>/dev/null) ]]
}

common_env=(
  PYTHONDONTWRITEBYTECODE=1
  PYTHONHASHSEED=0
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  PYTHONPATH="$workspace:$lingbot:$robotwin"
  LINGBOT_VA_MODEL="$model"
  LINGBOT_VA_STARTUP_SEED="${LINGBOT_VA_STARTUP_SEED:-0}"
  LINGBOT_VA_ENABLE_OFFLOAD="${LINGBOT_VA_ENABLE_OFFLOAD:-1}"
  LINGBOT_VA_CONSTRAINT_MODE="$constraint_mode"
  LINGBOT_VA_FEEDBACK_LIVE="$feedback_live"
  LINGBOT_VA_RTC_DELAY="${LINGBOT_VA_RTC_DELAY:-16}"
  LINGBOT_VA_RTC_EXECUTION_HORIZON="${LINGBOT_VA_RTC_EXECUTION_HORIZON:-16}"
  LINGBOT_VA_RTC_ATTENTION_SCHEDULE="${LINGBOT_VA_RTC_ATTENTION_SCHEDULE:-EXP}"
  LINGBOT_VA_FEEDBACK_OBS_PER_STATE="${LINGBOT_VA_FEEDBACK_OBS_PER_STATE:-4}"
  LINGBOT_VA_PSEUDO_VIDEO_SOLVER_STEPS="${LINGBOT_VA_PSEUDO_VIDEO_SOLVER_STEPS:-26}"
)

cd "$lingbot"
setsid env "${common_env[@]}" CUDA_VISIBLE_DEVICES="$server_gpu" \
  "$server_python" -u -m torch.distributed.run --nproc_per_node 1 \
  --master_port "$master_port" wan_va/wan_va_server.py --config-name robotwin \
  --port "$port" --save_root "$output/server_debug" \
  >"$output/logs/server.log" 2>&1 &
server_pid=$!

for _ in {1..180}; do
  port_is_listening && break
  kill -0 "$server_pid" 2>/dev/null || exit 3
  sleep 5
done
port_is_listening

# Rendering, CuRobo, and the policy server share the single RTX PRO 6000. The
# mathematical pseudo-clock is fixed and does not model their wall-clock latency.
env "${common_env[@]}" CUDA_VISIBLE_DEVICES="$client_gpu" \
  ROBOTWIN_ROOT="$robotwin" NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
  ROBOTWIN_RENDER_BACKEND=default SAPIEN_DISABLE_RAY_TRACING=1 \
  VK_ICD_FILENAMES="$vulkan_icd" \
  LD_LIBRARY_PATH="$client_site/sapien/oidn_library:$server_prefix/targets/x86_64-linux/lib:$server_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
  "$client_python" -u -m evaluation.robotwin.eval_polict_client_openpi \
  --config "$robotwin/policy/ACT/deploy_policy.yml" --overrides \
  --task_name "$task_name" --task_config "$task_config" --train_config_name 0 \
  --model_name 0 --ckpt_setting 0 --seed "$eval_seed" --policy_name ACT \
  --save_root "$output/client" --video_guidance_scale 5 \
  --action_guidance_scale 1 --test_num "$test_num" --port "$port" \
  >"$output/logs/client.log" 2>&1

st_seed=$((10000 * (1 + eval_seed)))
result="$output/client/stseed-${st_seed}/metrics/${task_name}/res.json"
[[ -s "$result" ]] || {
  echo "RoboTwin did not produce $result" >&2
  exit 4
}
"$client_python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["total_num"] == int(sys.argv[2])' \
  "$result" "$test_num"
printf '%s\n' "$output"
