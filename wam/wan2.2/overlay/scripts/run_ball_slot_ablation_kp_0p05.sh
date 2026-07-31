#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WAN22_PYTHON:-python}"
ckpt_dir="${WAN22_CKPT_DIR:?set WAN22_CKPT_DIR to the TI2V-5B checkpoint}"
result_dir="${repo_dir}/results/ball_meet_ball"
prompt="Static high-angle phone video of exactly two small rubber balls on a white office table. The orange miniature basketball rolls toward the stationary black-and-white miniature soccer ball and collides with it. The impact transfers momentum: the soccer ball rolls away, while the orange ball slows and rebounds slightly. The man beside the table remains still. Realistic physics, fixed camera, natural indoor lighting."

variants=(10 20 30)
release_schedules=(
  "1,3,4,6,8,9,11,12,14,16"
  "1,3,4,6,8,9,11,12,14,16,17,19,20,22,24,25,27,29,30,32"
  "1,3,4,6,8,9,11,12,14,16,17,19,20,22,24,25,27,29,30,32,33,35,37,38,40,41,43,45,46,48"
)

cd "${repo_dir}"
for index in "${!variants[@]}"; do
  slots="${variants[$index]}"
  output="${result_dir}/fbfm_kp_0p05_${slots}slots_future.mp4"
  audit="${result_dir}/fbfm_kp_0p05_${slots}slots_future.json"
  if [[ -s "${output}" && -s "${audit}" ]]; then
    echo "Skipping completed ${slots}-slot run"
    continue
  fi

  "${python_bin}" generate_fbfm.py \
    --mode FBFM \
    --ckpt-dir "${ckpt_dir}" \
    --image "${result_dir}/anchor_frame_048.png" \
    --feedback-video "${result_dir}/reference_future_121f.mp4" \
    --feedback-release-steps "${release_schedules[$index]}" \
    --prompt "${prompt}" \
    --size "1280*704" \
    --frame-num 121 \
    --sample-steps 50 \
    --sample-shift 5.0 \
    --guide-scale 5.0 \
    --beta 10.0 \
    --state-weight 1.0 \
    --kp 0.05 \
    --seed 0 \
    --output "${output}" \
    --audit "${audit}"
done
