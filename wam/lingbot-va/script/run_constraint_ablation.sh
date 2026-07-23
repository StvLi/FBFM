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

workspace=/mnt/project_eai_hs/zrm2/FBFM
lingbot="$workspace/wam/lingbot-va"
robotwin=${ROBOTWIN_ROOT:-/mnt/project_eai_hs/zrm/RoboTwin}
server_python=${LINGBOT_SERVER_PYTHON:-/mnt/project_eai_hs/zrm/miniconda3/envs/lingbot-va/bin/python}
client_python=${ROBOTWIN_CLIENT_PYTHON:-/mnt/project_eai_hs/zrm/venvs/robotwin-lingbot/bin/python}
model=${LINGBOT_VA_MODEL:-/mnt/project_eai_hs/zrm/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin}
port=${LINGBOT_VA_PORT:-29156}
master_port=${LINGBOT_VA_MASTER_PORT:-29161}
server_gpu=${LINGBOT_SERVER_GPU:-0}
client_gpu=${ROBOTWIN_CLIENT_GPU:-$(( (server_gpu + 1) % 4 ))}
[[ "$server_gpu" != "$client_gpu" ]] || {
  echo "policy server and RoboTwin client must use different GPUs" >&2
  exit 2
}
output=${LINGBOT_VA_ABLATION_ROOT:-$lingbot/robotwin_outputs}/adjust_bottle_${variant}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$output/logs" "$output/server_debug"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

common_env=(
  PYTHONDONTWRITEBYTECODE=1
  PYTHONHASHSEED=0
  PYTHONPATH="$workspace:$lingbot:$robotwin"
  LINGBOT_VA_MODEL="$model"
  LINGBOT_VA_ENABLE_OFFLOAD="${LINGBOT_VA_ENABLE_OFFLOAD:-1}"
  LINGBOT_VA_CONSTRAINT_MODE="$constraint_mode"
  LINGBOT_VA_FEEDBACK_LIVE="$feedback_live"
  LINGBOT_VA_RTC_DELAY="${LINGBOT_VA_RTC_DELAY:-16}"
  LINGBOT_VA_RTC_EXECUTION_HORIZON="${LINGBOT_VA_RTC_EXECUTION_HORIZON:-16}"
  LINGBOT_VA_RTC_ATTENTION_SCHEDULE="${LINGBOT_VA_RTC_ATTENTION_SCHEDULE:-EXP}"
)

cd "$lingbot"
setsid env "${common_env[@]}" CUDA_VISIBLE_DEVICES="$server_gpu" \
  "$server_python" -u -m torch.distributed.run --nproc_per_node 1 \
  --master_port "$master_port" wan_va/wan_va_server.py --config-name robotwin \
  --port "$port" --save_root "$output/server_debug" \
  >"$output/logs/server.log" 2>&1 &
server_pid=$!

for _ in {1..180}; do
  ss -ltn | rg -q ":${port} " && break
  kill -0 "$server_pid" 2>/dev/null || exit 3
  sleep 5
done
ss -ltn | rg -q ":${port} "

# Rendering uses llvmpipe, but RoboTwin's planner import still requires CUDA.
# Put that small CUDA context on a different physical GPU from the policy.
env "${common_env[@]}" CUDA_VISIBLE_DEVICES="$client_gpu" \
  ROBOTWIN_ROOT="$robotwin" NO_PROXY=127.0.0.1,localhost,0.0.0.0 \
  ROBOTWIN_RENDER_BACKEND=default SAPIEN_DISABLE_RAY_TRACING=1 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
  LD_LIBRARY_PATH="/mnt/project_eai_hs/zrm/venvs/robotwin-lingbot/lib/python3.10/site-packages/sapien/oidn_library:/mnt/project_eai_hs/zrm/miniconda3/envs/lingbot-va/targets/x86_64-linux/lib:/mnt/project_eai_hs/zrm/miniconda3/envs/lingbot-va/lib" \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
  "$client_python" -u -m evaluation.robotwin.eval_polict_client_openpi \
  --config "$robotwin/policy/ACT/deploy_policy.yml" --overrides \
  --task_name adjust_bottle --task_config demo_clean --train_config_name 0 \
  --model_name 0 --ckpt_setting 0 --seed 0 --policy_name ACT \
  --save_root "$output/client" --video_guidance_scale 5 \
  --action_guidance_scale 1 --test_num "$test_num" --port "$port" \
  >"$output/logs/client.log" 2>&1

result="$output/client/stseed-10000/metrics/adjust_bottle/res.json"
[[ -s "$result" ]] || {
  echo "RoboTwin did not produce $result" >&2
  exit 4
}
"$client_python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["total_num"] == int(sys.argv[2])' \
  "$result" "$test_num"
printf '%s\n' "$output"
