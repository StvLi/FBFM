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
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap/verify.sh [options]

Check pinned upstream revisions, licenses, FBFM patches, and optional model
paths.  This command is read-only and never launches an episode.

Options:
  --route ROUTE          all, lingbot, dreamzero, wan, or sim (default: all)
  --external-root PATH   upstream checkout directory (default: ./external)
  --checkpoints          require checkpoint/tokenizer paths as well as source
  --strict               treat missing optional sources/checkpoints as errors
  --dry-run              print checks without touching repositories
  -h, --help             show this help

Checkpoint overrides:
  LINGBOT_VA_MODEL, DREAMZERO_CHECKPOINT, DREAMZERO_TOKENIZER, WAN_CHECKPOINT
  (DREAMZERO_BASE_WORKSPACE is also honored.)
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
    --strict) STRICT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
done

case "$ROUTE" in
  all)       UPSTREAMS=(lingbot dreamzero robotwin libero rlinf wan) ;;
  lingbot)   UPSTREAMS=(lingbot robotwin) ;;
  dreamzero) UPSTREAMS=(dreamzero libero rlinf) ;;
  wan)       UPSTREAMS=(wan) ;;
  sim)       UPSTREAMS=(robotwin libero rlinf) ;;
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
  note "dry-run route=$ROUTE external_root=$EXTERNAL_ROOT strict=$STRICT checkpoints=$CHECKPOINTS"
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
      if [[ "$expected" == "MIT" ]]; then
        grep -Eiq 'MIT License|Permission is hereby granted, free of charge' "$dir/$file" \
          || warn "$key license file $file does not contain the expected MIT notice"
      else
        grep -Eiq 'Apache License|Licensed under the Apache License' "$dir/$file" \
          || warn "$key license file $file does not contain the expected Apache notice"
      fi
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
done

# A patch is valid against a clean pinned Wan checkout when either applying it
# or reversing it succeeds.  The reverse case means the user already applied
# the overlay and is still reproducible.
wan_dir="$EXTERNAL_ROOT/${DEST[wan]}"
patch_candidates=(
  "$REPO_ROOT/third_party/patches/wan2.2_fbfm.patch"
  "$REPO_ROOT/wam/wan2.2/patches/wan2.2_fbfm.patch"
)
wan_patch=""
for candidate in "${patch_candidates[@]}"; do
  [[ -f "$candidate" ]] && { wan_patch="$candidate"; break; }
done
if [[ -n "$wan_patch" && -d "$wan_dir" && -e "$wan_dir/.git" ]]; then
  if git -C "$wan_dir" apply --check "$wan_patch" >/dev/null 2>&1; then
    note "Wan2.2 FBFM patch applies cleanly"
  elif git -C "$wan_dir" apply --reverse --check "$wan_patch" >/dev/null 2>&1; then
    note "Wan2.2 FBFM patch is already applied"
  else
    error "Wan2.2 FBFM patch does not apply to $wan_dir"
  fi
elif [[ "$ROUTE" == wan || "$ROUTE" == all ]]; then
  required_or_warn "Wan2.2 FBFM patch not found (expected third_party/patches/wan2.2_fbfm.patch)"
fi

check_path() {
  local label="$1" path="$2"
  if [[ -d "$path" || -f "$path" ]]; then
    note "$label: $path"
  else
    required_or_warn "missing $label: $path"
  fi
}

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
  [[ -n "$checkpoint" ]] || [[ -z "$base" ]] || checkpoint="$base/checkpoints/RLinf-DreamZero-WAN2.2-5B-LIBERO-SFT-Step26000"
  [[ -n "$tokenizer" ]] || [[ -z "$base" ]] || tokenizer="$base/assets/tokenizers/umt5-xxl"
  if [[ -n "$checkpoint" ]]; then check_path "DreamZero checkpoint" "$checkpoint"; elif ((CHECKPOINTS)); then required_or_warn "set DREAMZERO_CHECKPOINT or DREAMZERO_BASE_WORKSPACE"; else note "DreamZero checkpoint not checked"; fi
  if [[ -n "$tokenizer" ]]; then check_path "UMT5 tokenizer" "$tokenizer"; elif ((CHECKPOINTS)); then required_or_warn "set DREAMZERO_TOKENIZER or DREAMZERO_BASE_WORKSPACE"; else note "UMT5 tokenizer not checked"; fi
fi

if [[ "$ROUTE" == wan || "$ROUTE" == all ]]; then
  wan_checkpoint="${WAN_CHECKPOINT:-}"
  if [[ -n "$wan_checkpoint" ]]; then check_path "Wan2.2 checkpoint" "$wan_checkpoint"; elif ((CHECKPOINTS)); then required_or_warn "set WAN_CHECKPOINT to check the Wan model directory"; else note "Wan2.2 checkpoint not checked"; fi
fi

note "verification finished: $errors error(s), $warnings warning(s)"
if ((errors > 0)); then exit 1; fi
exit 0
