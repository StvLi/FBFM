#!/usr/bin/env bash
# Fetch the pinned upstream repositories used by the three FBFM routes.
#
# This script deliberately keeps upstream code outside the Git checkout.  It is
# safe to re-run: an existing clean checkout is moved to the pinned revision,
# while a checkout containing local changes is never overwritten.  Checkpoints
# and datasets are intentionally not downloaded here.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
EXTERNAL_ROOT="${FBFM_EXTERNAL_ROOT:-$REPO_ROOT/external}"
ROUTE="all"
DRY_RUN=0
OFFLINE=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap/fetch_upstreams.sh [options]

Fetch upstreams at the revisions used by the paper.  Checked-in integration
patches are applied only after each pinned revision is verified.

Options:
  --route ROUTE          all, lingbot, dreamzero, wan, or sim (default: all)
  --external-root PATH   destination for upstream checkouts (default: ./external)
  --dry-run              print actions without changing the filesystem
  --offline              do not contact a remote; require the revision locally
  -h, --help             show this help

The FBFM_EXTERNAL_ROOT environment variable is an equivalent way to set the
external root.  Existing dirty checkouts are never reset, cleaned, or forced.
EOF
}

die() {
  echo "fetch_upstreams: error: $*" >&2
  exit 1
}

log() {
  echo "fetch_upstreams: $*"
}

while (($#)); do
  case "$1" in
    --route)
      (($# >= 2)) || die "--route requires a value"
      ROUTE="$2"
      shift 2
      ;;
    --route=*) ROUTE="${1#*=}"; shift ;;
    --external-root)
      (($# >= 2)) || die "--external-root requires a path"
      EXTERNAL_ROOT="$2"
      shift 2
      ;;
    --external-root=*) EXTERNAL_ROOT="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --offline|--no-fetch) OFFLINE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
done

case "$ROUTE" in
  all)      UPSTREAMS=(lingbot dreamzero robotwin pytorch3d curobo libero rlinf wan) ;;
  lingbot)  UPSTREAMS=(lingbot robotwin pytorch3d curobo) ;;
  dreamzero) UPSTREAMS=(dreamzero libero rlinf) ;;
  wan)      UPSTREAMS=(wan) ;;
  sim)      UPSTREAMS=(robotwin pytorch3d curobo libero rlinf) ;;
  *) die "unsupported route '$ROUTE'; choose all, lingbot, dreamzero, wan, or sim" ;;
esac

# Bash 4 associative arrays keep the manifest in one auditable place.
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

if ((DRY_RUN == 0)); then
  mkdir -p "$EXTERNAL_ROOT"
fi

run() {
  if ((DRY_RUN)); then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

repo_dirty() {
  [[ -n "$(git -C "$1" status --porcelain --untracked-files=normal 2>/dev/null)" ]]
}

patched_tree_matches() {
  local dest="$1" patch="$2" tmpdir index extra
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/fbfm-patch-index.XXXXXX")"
  index="$tmpdir/index"
  if ! GIT_INDEX_FILE="$index" git -C "$dest" read-tree HEAD \
    || ! GIT_INDEX_FILE="$index" git -C "$dest" apply --cached "$patch" \
    || ! GIT_INDEX_FILE="$index" git -C "$dest" diff --quiet; then
    rm -rf -- "$tmpdir"
    return 1
  fi
  extra="$(GIT_INDEX_FILE="$index" git -C "$dest" ls-files --others --exclude-standard)"
  rm -rf -- "$tmpdir"
  [[ -z "$extra" ]]
}

apply_integration_patch() {
  local key="$1" label="$2" patch="$3"
  local dest="$EXTERNAL_ROOT/${DEST[$key]}"
  [[ -f "$patch" ]] || die "$label patch is missing: $patch"

  if git -C "$dest" apply --reverse --check "$patch" >/dev/null 2>&1; then
    patched_tree_matches "$dest" "$patch" \
      || die "$dest contains the $label patch plus other local changes; refusing to continue"
    log "$label patch already applied"
    return 0
  fi
  repo_dirty "$dest" \
    && die "$dest has local changes unrelated to the $label patch; refusing to apply it"
  git -C "$dest" apply --check "$patch" \
    || die "$label patch does not apply to pinned revision ${REV[$key]}"
  git -C "$dest" apply "$patch"
  patched_tree_matches "$dest" "$patch" \
    || die "$label checkout differs from the expected patched tree after apply"
  log "applied $label patch"
}

ensure_upstream() {
  local key="$1"
  local dest="$EXTERNAL_ROOT/${DEST[$key]}"
  local url="${URL[$key]}"
  local rev="${REV[$key]}"

  log "$key -> $dest @ ${rev:0:12}"
  if ((DRY_RUN)); then
    if [[ -e "$dest" ]]; then
      run git -C "$dest" fetch --tags origin
      run git -C "$dest" checkout --detach "$rev"
    else
      run git clone "$url" "$dest"
      run git -C "$dest" checkout --detach "$rev"
    fi
    return 0
  fi

  if [[ -e "$dest" && ! -d "$dest" ]]; then
    die "destination exists but is not a directory: $dest"
  fi

  if [[ ! -d "$dest/.git" && ! -f "$dest/.git" ]]; then
    ((OFFLINE == 0)) || die "$key is absent and --offline was requested"
    mkdir -p "$(dirname -- "$dest")"
    git clone "$url" "$dest"
  elif [[ ! -d "$dest/.git" && -f "$dest/.git" ]]; then
    # Worktrees and submodules use a .git file; git -C handles both.
    git -C "$dest" rev-parse --git-dir >/dev/null 2>&1 \
      || die "$dest has a .git file but is not a valid checkout"
  fi

  git -C "$dest" rev-parse --show-toplevel >/dev/null 2>&1 \
    || die "$dest is not a Git checkout"

  local current=""
  current="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current" != "$rev" ]]; then
    if repo_dirty "$dest"; then
      die "$dest has local changes; refusing to change revision (commit/stash them and retry)"
    fi
    if ((OFFLINE == 0)); then
      # Fetch tags/branches first; if the server allows SHA wants, fetch the
      # exact object as a fallback.  Existing local objects make this a no-op.
      git -C "$dest" fetch --tags origin || log "warning: unable to fetch tags for $key; checking local objects"
      if ! git -C "$dest" cat-file -e "$rev^{commit}" 2>/dev/null; then
        git -C "$dest" fetch origin "$rev" \
          || die "could not fetch pinned revision $rev for $key"
      fi
    fi
    git -C "$dest" cat-file -e "$rev^{commit}" 2>/dev/null \
      || die "pinned revision $rev for $key is not available (remove --offline or fetch it first)"
    git -C "$dest" checkout --detach "$rev"
  fi

  [[ "$(git -C "$dest" rev-parse HEAD)" == "$rev" ]] \
    || die "$key checkout did not land on $rev"
}

for key in "${UPSTREAMS[@]}"; do
  ensure_upstream "$key"
  case "$key" in
    robotwin)
      patch="$REPO_ROOT/wam/lingbot-va/patches/robotwin_raster_backend.patch"
      label="RoboTwin raster-backend"
      ;;
    dreamzero)
      patch="$REPO_ROOT/wam/dreamzero-libero/patches/dreamzero_external_step_guidance.patch"
      label="DreamZero scheduler-callback"
      ;;
    wan)
      patch="$REPO_ROOT/wam/wan2.2/patches/wan2.2_fbfm.patch"
      label="Wan2.2 FBFM"
      ;;
    *)
      if ((DRY_RUN == 0)) && repo_dirty "$EXTERNAL_ROOT/${DEST[$key]}"; then
        die "$EXTERNAL_ROOT/${DEST[$key]} has local changes; expected a clean pinned checkout"
      fi
      continue
      ;;
  esac
  if ((DRY_RUN)); then
    run git -C "$EXTERNAL_ROOT/${DEST[$key]}" apply "$patch"
  else
    apply_integration_patch "$key" "$label" "$patch"
  fi
done

if ((DRY_RUN == 0)); then
  lock="$EXTERNAL_ROOT/manifest.lock"
  tmp="$lock.tmp.$$"
  {
    printf '# Generated by scripts/bootstrap/fetch_upstreams.sh\n'
    printf '# route=%s generated_at=%s\n' "$ROUTE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for key in "${UPSTREAMS[@]}"; do
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$key" "${DEST[$key]}" "${URL[$key]}" "${REV[$key]}" "${LICENSE[$key]}"
    done
  } >"$tmp"
  mv -f -- "$tmp" "$lock"
  log "wrote $lock"
else
  log "dry-run complete; no files changed"
fi
