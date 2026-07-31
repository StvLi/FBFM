#!/usr/bin/env bash
# Verify source revisions and deployment prerequisites without starting a model
# or simulator.  Missing external sources are warnings by default so that a
# freshly cloned submission can explain what fetch_upstreams.sh must provide;
# pass --strict for a CI/deployment gate.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
EXTERNAL_ROOT="${FBFM_EXTERNAL_ROOT:-$REPO_ROOT/external}"
ROUTE="all"
STRICT=0
CHECKPOINTS=0
ASSETS=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap/verify.sh [options]

Check pinned upstream revisions, licenses, FBFM patches, and optional model
component paths.  This command is read-only and never launches an episode.

Options:
  --route ROUTE          all, lingbot, dreamzero, wan, or sim (default: all)
  --external-root PATH   upstream checkout directory (default: ./external)
  --checkpoints          require all route model artifacts as well as source
  --assets               require configured RoboTwin simulator assets
  --strict               treat missing optional sources/artifacts as errors
  --dry-run              print checks without touching repositories
  -h, --help             show this help

Checkpoint overrides:
  LINGBOT_VA_MODEL, DREAMZERO_CHECKPOINT, DREAMZERO_TOKENIZER,
  DREAMZERO_WAN_CHECKPOINT, DREAMZERO_IMAGE_ENCODER, WAN_CHECKPOINT
  (WAN22_CKPT_DIR is also accepted for the Wan visual-only route.)

  DREAMZERO_BASE_WORKSPACE is also honored for the SFT checkpoint/tokenizer.
EOF
}

die() {
  echo "verify: error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --route)
      (($# >= 2)) || die "--route requires a value"
      ROUTE="$2"; shift 2 ;;
    --route=*) ROUTE="${1#*=}"; shift ;;
    --external-root)
      (($# >= 2)) || die "--external-root requires a path"
      EXTERNAL_ROOT="$2"; shift 2 ;;
    --external-root=*) EXTERNAL_ROOT="${1#*=}"; shift ;;
    --checkpoints) CHECKPOINTS=1; shift ;;
    --assets) ASSETS=1; shift ;;
    --strict) STRICT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
done

case "$ROUTE" in
  all)       UPSTREAMS=(lingbot dreamzero robotwin pytorch3d curobo libero rlinf wan) ;;
  lingbot)   UPSTREAMS=(lingbot robotwin pytorch3d curobo) ;;
  dreamzero) UPSTREAMS=(dreamzero libero rlinf) ;;
  wan)       UPSTREAMS=(wan) ;;
  sim)       UPSTREAMS=(robotwin pytorch3d curobo libero rlinf) ;;
  *) die "unsupported route '$ROUTE'; choose all, lingbot, dreamzero, wan, or sim" ;;
esac

declare -A URL REV DEST LICENSE
URL[lingbot]="https://github.com/robbyant/lingbot-va.git"
REV[lingbot]="7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
DEST[lingbot]="lingbot-va"
LICENSE[lingbot]="Apache-2.0"
URL[dreamzero]="https://github.com/dreamzero0/dreamzero.git"
REV[dreamzero]="ab790c198fbce33503358efbbd4187ce9a89adf3"
DEST[dreamzero]="dreamzero"
LICENSE[dreamzero]="Apache-2.0"
URL[robotwin]="https://github.com/RoboTwin-Platform/RoboTwin.git"
REV[robotwin]="2eeec322d95799f537cbfe5f291a8220d965ccb8"
DEST[robotwin]="RoboTwin"
LICENSE[robotwin]="MIT"
URL[pytorch3d]="https://github.com/facebookresearch/pytorch3d.git"
REV[pytorch3d]="32a33e24428d07171ef54e359d902205eab95b9b"
DEST[pytorch3d]="pytorch3d"
LICENSE[pytorch3d]="BSD-3-Clause"
URL[curobo]="https://github.com/NVlabs/curobo.git"
REV[curobo]="0db44e5916492ad814baf2764b88cc156d22e525"
DEST[curobo]="curobo"
LICENSE[curobo]="NVIDIA-Source-Code-License"
URL[libero]="https://github.com/RLinf/LIBERO.git"
REV[libero]="0c5e40cc4ae63e09c14e7df6f74481e9ee8585f7"
DEST[libero]="LIBERO"
LICENSE[libero]="MIT"
URL[rlinf]="https://github.com/RLinf/RLinf.git"
REV[rlinf]="26179807d701950cf2933554bfb9bb596e662b68"
DEST[rlinf]="RLinf"
LICENSE[rlinf]="Apache-2.0"
URL[wan]="https://github.com/Wan-Video/Wan2.2.git"
REV[wan]="42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
DEST[wan]="Wan2.2"
LICENSE[wan]="Apache-2.0"

errors=0
warnings=0
note() { echo "verify: $*"; }
warn() { echo "verify: warning: $*" >&2; warnings=$((warnings + 1)); }
error() { echo "verify: error: $*" >&2; errors=$((errors + 1)); }

required_or_warn() {
  if ((STRICT)); then error "$*"; else warn "$*"; fi
}

if ((DRY_RUN)); then
  note "dry-run route=$ROUTE external_root=$EXTERNAL_ROOT strict=$STRICT checkpoints=$CHECKPOINTS assets=$ASSETS"
  for key in "${UPSTREAMS[@]}"; do
    note "would check $EXTERNAL_ROOT/${DEST[$key]} @ ${REV[$key]} (${LICENSE[$key]})"
  done
  exit 0
fi

check_license() {
  local key="$1" dir="$2" expected="${LICENSE[$1]}" file found=0
  for file in LICENSE LICENSE.txt COPYING COPYING.txt; do
    if [[ -f "$dir/$file" ]]; then
      found=1
      case "$expected" in
        MIT)
          grep -Eiq 'MIT License|Permission is hereby granted, free of charge' "$dir/$file" \
            || warn "$key license file $file does not contain the expected MIT notice"
          ;;
        Apache-2.0)
          grep -Eiq 'Apache License|Licensed under the Apache License' "$dir/$file" \
            || warn "$key license file $file does not contain the expected Apache notice"
          ;;
        BSD-3-Clause)
          grep -Eiq 'Redistribution and use in source and binary forms' "$dir/$file" \
            || warn "$key license file $file does not contain the expected BSD notice"
          ;;
        NVIDIA-Source-Code-License)
          grep -Eiq 'NVIDIA License' "$dir/$file" \
            && grep -Eiq 'non-commercially' "$dir/$file" \
            || warn "$key license file $file does not contain the expected NVIDIA terms"
          ;;
        *) warn "$key has an unsupported expected license identifier: $expected" ;;
      esac
      break
    fi
  done
  ((found)) || required_or_warn "$key has no top-level LICENSE/COPYING file"
}

for key in "${UPSTREAMS[@]}"; do
  dir="$EXTERNAL_ROOT/${DEST[$key]}"
  if [[ ! -d "$dir" ]]; then
    required_or_warn "missing upstream checkout: $dir (run fetch_upstreams.sh)"
    continue
  fi
  if [[ ! -e "$dir/.git" ]]; then
    error "$dir is not a Git checkout"
    continue
  fi
  head="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$head" != "${REV[$key]}" ]]; then
    error "$key revision mismatch: expected ${REV[$key]}, found ${head:-<none>}"
  else
    note "$key revision ${head:0:12} OK"
  fi
  check_license "$key" "$dir"
  case "$key" in
    robotwin|dreamzero|wan) ;;
    *)
      if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
        error "$key checkout has local changes; expected a clean pinned tree"
      else
        note "$key working tree clean"
      fi
      ;;
  esac
done

# Check patches against each pinned Git index.  --cached deliberately ignores
# overlay files already present in a working tree, so this remains read-only and
# works both before and after fetch_upstreams.sh applies the patch.
is_selected() {
  local wanted="$1" selected
  for selected in "${UPSTREAMS[@]}"; do
    [[ "$selected" != "$wanted" ]] || return 0
  done
  return 1
}

check_integration_patch() {
  local key="$1" label="$2" patch="$3"
  local dir="$EXTERNAL_ROOT/${DEST[$key]}"
  is_selected "$key" || return 0
  if [[ ! -f "$patch" ]]; then
    required_or_warn "$label patch not found: $patch"
    return 0
  fi
  [[ -d "$dir" && -e "$dir/.git" ]] || return 0
  if git -C "$dir" apply --reverse --check "$patch" >/dev/null 2>&1; then
    local scratch index extras
    scratch="$(mktemp -d "${TMPDIR:-/tmp}/fbfm-verify-index.XXXXXX")"
    index="$scratch/index"
    if GIT_INDEX_FILE="$index" git -C "$dir" read-tree HEAD \
      && GIT_INDEX_FILE="$index" git -C "$dir" apply --cached "$patch" \
      && GIT_INDEX_FILE="$index" git -C "$dir" diff --quiet \
      && [[ -z "$(GIT_INDEX_FILE="$index" git -C "$dir" ls-files --others --exclude-standard)" ]]; then
      note "$label patch is already applied and the working tree matches it exactly"
    else
      error "$label checkout contains the patch plus unrelated local changes"
    fi
    rm -rf -- "$scratch"
  elif [[ -n "$(git -C "$dir" status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
    error "$label checkout has local changes that do not match the reviewed patch"
  elif git -C "$dir" apply --check --cached "$patch" >/dev/null 2>&1; then
    if ((STRICT)); then
      error "$label checkout is pristine but the required patch is not applied; run fetch_upstreams.sh to apply it"
    else
      note "$label checkout is pristine; the required patch is not applied but applies cleanly"
    fi
  else
    error "$label patch does not apply to $dir"
  fi
}

check_integration_patch robotwin "RoboTwin raster-backend" \
  "$REPO_ROOT/wam/lingbot-va/patches/robotwin_raster_backend.patch"
check_integration_patch dreamzero "DreamZero scheduler-callback" \
  "$REPO_ROOT/wam/dreamzero-libero/patches/dreamzero_external_step_guidance.patch"
check_integration_patch wan "Wan2.2 FBFM" \
  "$REPO_ROOT/wam/wan2.2/patches/wan2.2_fbfm.patch"

check_path() {
  local label="$1" path="$2"
  if [[ -d "$path" || -f "$path" ]]; then
    note "$label: $path"
  else
    required_or_warn "missing $label: $path"
  fi
}

check_directory() {
  local label="$1" path="$2"
  if [[ -d "$path" ]]; then
    note "$label: $path"
  else
    required_or_warn "missing $label directory: $path"
  fi
}

check_file() {
  local label="$1" path="$2"
  if [[ -f "$path" ]]; then
    note "$label: $path"
  else
    required_or_warn "missing $label file: $path"
  fi
}

if ((ASSETS)) && is_selected robotwin; then
  robotwin_root="$EXTERNAL_ROOT/${DEST[robotwin]}"
  asset_python="${FBFM_ROBOTWIN_PYTHON:-${FBFM_ENV_ROOT:-$REPO_ROOT/.venvs}/fbfm-robotwin/bin/python}"
  if [[ ! -x "$asset_python" ]]; then
    asset_python="$(command -v python3 || true)"
  fi
  if [[ -z "$asset_python" ]]; then
    error "no Python interpreter is available for the RoboTwin asset preflight"
  elif "$asset_python" "$REPO_ROOT/scripts/bootstrap/fetch_robotwin_assets.py" \
      --robotwin-root "$robotwin_root" --check; then
    note "RoboTwin pinned assets and configured embodiment paths OK"
  else
    error "RoboTwin asset preflight failed; run fetch_robotwin_assets.py"
  fi
elif ((ASSETS)); then
  note "--assets requested, but the selected route does not use RoboTwin"
fi

if [[ "$ROUTE" == lingbot || "$ROUTE" == all ]]; then
  lingbot_model="${LINGBOT_VA_MODEL:-${FBFM_LINGBOT_VA_MODEL:-}}"
  if [[ -n "$lingbot_model" ]]; then
    check_path "LingBot-VA checkpoint" "$lingbot_model"
  elif ((CHECKPOINTS)); then
    required_or_warn "set LINGBOT_VA_MODEL to the LingBot-VA checkpoint directory"
  else
    note "LingBot-VA checkpoint not checked (set LINGBOT_VA_MODEL to check it)"
  fi
fi

if [[ "$ROUTE" == dreamzero || "$ROUTE" == all ]]; then
  base="${DREAMZERO_BASE_WORKSPACE:-}"
  checkpoint="${DREAMZERO_CHECKPOINT:-}"
  tokenizer="${DREAMZERO_TOKENIZER:-}"
  dreamzero_wan_checkpoint="${DREAMZERO_WAN_CHECKPOINT:-}"
  dreamzero_image_encoder="${DREAMZERO_IMAGE_ENCODER:-}"
  [[ -n "$checkpoint" ]] || [[ -z "$base" ]] || checkpoint="$base/checkpoints/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000"
  [[ -n "$tokenizer" ]] || [[ -z "$base" ]] || tokenizer="$base/assets/tokenizers/umt5-xxl"
  if [[ -n "$checkpoint" ]]; then check_path "DreamZero checkpoint" "$checkpoint"; elif ((CHECKPOINTS)); then required_or_warn "set DREAMZERO_CHECKPOINT or DREAMZERO_BASE_WORKSPACE"; else note "DreamZero checkpoint not checked"; fi
  if [[ -n "$tokenizer" ]]; then check_path "UMT5 tokenizer" "$tokenizer"; elif ((CHECKPOINTS)); then required_or_warn "set DREAMZERO_TOKENIZER or DREAMZERO_BASE_WORKSPACE"; else note "UMT5 tokenizer not checked"; fi
  if [[ -n "$dreamzero_wan_checkpoint" ]]; then
    check_directory "DreamZero Wan2.2 checkpoint" "$dreamzero_wan_checkpoint"
    check_file "DreamZero Wan text encoder" \
      "$dreamzero_wan_checkpoint/models_t5_umt5-xxl-enc-bf16.pth"
    check_file "DreamZero Wan2.2 VAE" \
      "$dreamzero_wan_checkpoint/Wan2.2_VAE.pth"
  elif ((CHECKPOINTS)); then
    required_or_warn \
      "set DREAMZERO_WAN_CHECKPOINT to the Wan2.2-TI2V-5B directory"
  else
    note "DreamZero Wan2.2 components not checked"
  fi
  if [[ -n "$dreamzero_image_encoder" ]]; then
    check_file "DreamZero CLIP image encoder" "$dreamzero_image_encoder"
  elif ((CHECKPOINTS)); then
    required_or_warn \
      "set DREAMZERO_IMAGE_ENCODER to models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  else
    note "DreamZero CLIP image encoder not checked"
  fi
fi

if [[ "$ROUTE" == wan || "$ROUTE" == all ]]; then
  wan_checkpoint="${WAN_CHECKPOINT:-${WAN22_CKPT_DIR:-}}"
  if [[ -n "$wan_checkpoint" ]]; then
    check_directory "Wan2.2 checkpoint" "$wan_checkpoint"
    if [[ -d "$wan_checkpoint" ]]; then
      check_file "Wan2.2 model config" "$wan_checkpoint/config.json"
      check_file "Wan2.2 diffusion index" \
        "$wan_checkpoint/diffusion_pytorch_model.safetensors.index.json"
      for shard in 00001-of-00003 00002-of-00003 00003-of-00003; do
        check_file "Wan2.2 diffusion shard $shard" \
          "$wan_checkpoint/diffusion_pytorch_model-$shard.safetensors"
      done
      check_file "Wan2.2 T5 encoder" \
        "$wan_checkpoint/models_t5_umt5-xxl-enc-bf16.pth"
      check_file "Wan2.2 VAE" "$wan_checkpoint/Wan2.2_VAE.pth"
      tokenizer_dir="$wan_checkpoint/google/umt5-xxl"
      check_file "Wan2.2 tokenizer config" "$tokenizer_dir/tokenizer_config.json"
      check_file "Wan2.2 tokenizer JSON" "$tokenizer_dir/tokenizer.json"
      check_file "Wan2.2 tokenizer SentencePiece model" "$tokenizer_dir/spiece.model"
      check_file "Wan2.2 tokenizer special tokens" \
        "$tokenizer_dir/special_tokens_map.json"
    fi
  elif ((CHECKPOINTS)); then
    required_or_warn \
      "set WAN_CHECKPOINT or WAN22_CKPT_DIR to the Wan2.2-TI2V-5B directory"
  else
    note "Wan2.2 checkpoint not checked"
  fi
fi

note "verification finished: $errors error(s), $warnings warning(s)"
if ((errors > 0)); then exit 1; fi
exit 0
