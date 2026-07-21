#!/usr/bin/env bash
set -euo pipefail

MODE=${FBFM_CONSTRAINT_MODE:-None}
case "$MODE" in None|RTC|Feedback) ;; *) echo "Invalid FBFM_CONSTRAINT_MODE=$MODE" >&2; exit 2;; esac

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the post-trained DreamZero-RoboTwin checkpoint}
ROBOTWIN_SCHEMA=${ROBOTWIN_SCHEMA:-$MODEL_PATH/robotwin_schema.json}
CHECKPOINT_MANIFEST=${CHECKPOINT_MANIFEST:-$MODEL_PATH/checkpoint_manifest.json}
PORT=${PORT:-29500}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES

for required in "$ROBOTWIN_SCHEMA" "$CHECKPOINT_MANIFEST" "$MODEL_PATH/experiment_cfg/conf.yaml"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done

python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
  --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --constraint-mode "$MODE" \
  --robotwin-schema "$ROBOTWIN_SCHEMA" \
  --checkpoint-manifest "$CHECKPOINT_MANIFEST"
