#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR=1

ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:?Set ROBOTWIN_DATA_ROOT to the native RoboTwin LeRobot dataset}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/dreamzero_robotwin_lora}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-./checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-./checkpoints/umt5-xxl}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-./checkpoints/DreamZero-AgiBot}
NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-1000}
REPORT_TO=${REPORT_TO:-none}
WANDB_PROJECT=${WANDB_PROJECT:-dreamzero_robotwin}

for required in \
  "$ROBOTWIN_DATA_ROOT/meta/embodiment.json" \
  "$ROBOTWIN_DATA_ROOT/meta/modality.json" \
  "$ROBOTWIN_DATA_ROOT/meta/stats.json" \
  "$ROBOTWIN_DATA_ROOT/meta/relative_stats_dreamzero.json" \
  "$PRETRAINED_MODEL_PATH/model.safetensors.index.json" \
  "$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
  "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  "$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required file is missing: $required" >&2
    exit 1
  fi
done

python -m evaluation.robotwin.validate_dataset "$ROBOTWIN_DATA_ROOT"

torchrun --nproc_per_node "$NUM_GPUS" --standalone groot/vla/experiment/experiment.py \
  report_to="$REPORT_TO" \
  data=dreamzero/robotwin_relative \
  wandb_project="$WANDB_PROJECT" \
  train_architecture=lora \
  num_frames=33 \
  action_horizon=24 \
  num_views=3 \
  model=dreamzero/vla \
  model/dreamzero/action_head=wan_flow_matching_action_tf \
  model/dreamzero/transform=dreamzero_cotrain \
  num_frame_per_block=2 \
  num_action_per_block=24 \
  num_state_per_block=1 \
  seed=42 \
  training_args.learning_rate=1e-5 \
  training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json \
  save_steps="$SAVE_STEPS" \
  training_args.warmup_ratio=0.05 \
  output_dir="$OUTPUT_DIR" \
  per_device_train_batch_size=1 \
  max_steps="$MAX_STEPS" \
  weight_decay=1e-5 \
  save_total_limit=10 \
  upload_checkpoints=false \
  bf16=true \
  tf32=true \
  eval_bf16=true \
  dataloader_pin_memory=false \
  dataloader_num_workers=4 \
  image_resolution_width=320 \
  image_resolution_height=176 \
  save_lora_only=true \
  max_chunk_size=4 \
  frame_seqlen=880 \
  save_strategy=steps \
  robotwin_data_root="$ROBOTWIN_DATA_ROOT" \
  dit_version="$WAN_CKPT_DIR" \
  text_encoder_pretrained_path="$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
  image_encoder_pretrained_path="$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  vae_pretrained_path="$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
  tokenizer_path="$TOKENIZER_DIR" \
  pretrained_model_path="$PRETRAINED_MODEL_PATH" \
  ++action_head_cfg.config.skip_component_loading=true \
  ++action_head_cfg.config.defer_lora_injection=true
