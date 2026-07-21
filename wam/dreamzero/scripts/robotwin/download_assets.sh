#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/mnt/project_eai_hs/zrm/checkpoints/dreamzero}
mkdir -p "$CHECKPOINT_ROOT"

download() {
  local repo=$1
  local destination=$2
  if [[ -d "$destination" && -n "$(find "$destination" -type f -print -quit 2>/dev/null)" ]]; then
    echo "Already present: $destination"
    return
  fi
  hf download "$repo" --repo-type model --local-dir "$destination"
}

download GEAR-Dreams/DreamZero-AgiBot "$CHECKPOINT_ROOT/DreamZero-AgiBot"
download Wan-AI/Wan2.1-I2V-14B-480P "$CHECKPOINT_ROOT/Wan2.1-I2V-14B-480P"
download google/umt5-xxl "$CHECKPOINT_ROOT/umt5-xxl"
