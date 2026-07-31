#!/usr/bin/env bash
# Recreate the pinned Wan2.2 checkout and apply only the reviewed FBFM patch.
set -Eeuo pipefail
IFS=$'\n\t'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
route_dir="$(cd -- "${script_dir}/.." && pwd -P)"
lock_file="${route_dir}/UPSTREAM.lock"
offline=0
upstream_dir=""

usage() {
  cat <<'EOF'
Usage: wam/wan2.2/scripts/fetch_upstream.sh [options] [DESTINATION]

Clone the Wan2.2 revision recorded in UPSTREAM.lock and apply the FBFM patch.
The default destination is wam/wan2.2/vendor/Wan2.2.

Options:
  --offline    never contact the remote; require an existing local checkout
  -h, --help   show this help

An already patched checkout is accepted only when its complete tracked and
untracked working tree exactly matches the reviewed patch. Other local changes
are never reset, cleaned, or overwritten.
EOF
}

die() {
  echo "fetch_upstream: error: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --offline|--no-fetch) offline=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option '$1' (use --help)" ;;
    *)
      [[ -z "$upstream_dir" ]] || die "only one destination may be supplied"
      upstream_dir="$1"
      shift
      ;;
  esac
done
upstream_dir="${upstream_dir:-${route_dir}/vendor/Wan2.2}"

lock_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; found=1; exit }
    END { if (!found) exit 1 }' "$lock_file"
}

[[ -f "$lock_file" ]] || die "missing lock file: $lock_file"
upstream_url="$(lock_value upstream_repository)" \
  || die "UPSTREAM.lock has no upstream_repository"
upstream_commit="$(lock_value upstream_commit)" \
  || die "UPSTREAM.lock has no upstream_commit"
upstream_branch="$(lock_value upstream_branch)" \
  || die "UPSTREAM.lock has no upstream_branch"
patch_relative="$(lock_value overlay_patch)" \
  || die "UPSTREAM.lock has no overlay_patch"
[[ "$upstream_commit" =~ ^[0-9a-f]{40}$ ]] \
  || die "invalid upstream_commit in UPSTREAM.lock"
[[ "$patch_relative" != /* && "$patch_relative" != *".."* ]] \
  || die "overlay_patch must be a route-relative path"
patch_file="${route_dir}/${patch_relative}"
[[ -f "$patch_file" ]] || die "missing FBFM patch: $patch_file"

repo_dirty() {
  [[ -n "$(git -C "$1" status --porcelain --untracked-files=normal 2>/dev/null)" ]]
}

# Compare the whole working tree with HEAD + patch using a throw-away index.
# This distinguishes the expected uncommitted overlay from unrelated edits.
patched_tree_matches() {
  local destination="$1" scratch index extras
  scratch="$(mktemp -d "${TMPDIR:-/tmp}/fbfm-wan-index.XXXXXX")"
  index="$scratch/index"
  if ! GIT_INDEX_FILE="$index" git -C "$destination" read-tree HEAD \
    || ! GIT_INDEX_FILE="$index" git -C "$destination" apply --cached "$patch_file" \
    || ! GIT_INDEX_FILE="$index" git -C "$destination" diff --quiet; then
    rm -rf -- "$scratch"
    return 1
  fi
  extras="$(GIT_INDEX_FILE="$index" git -C "$destination" \
    ls-files --others --exclude-standard)"
  rm -rf -- "$scratch"
  [[ -z "$extras" ]]
}

if [[ -e "$upstream_dir" && ! -d "$upstream_dir" ]]; then
  die "destination exists but is not a directory: $upstream_dir"
fi
if [[ ! -e "$upstream_dir/.git" ]]; then
  [[ ! -e "$upstream_dir" || -z "$(find "$upstream_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] \
    || die "destination exists but is not a Git checkout: $upstream_dir"
  ((offline == 0)) || die "checkout is absent and --offline was requested"
  mkdir -p "$(dirname -- "$upstream_dir")"
  git clone "$upstream_url" "$upstream_dir"
fi
git -C "$upstream_dir" rev-parse --show-toplevel >/dev/null 2>&1 \
  || die "destination is not a Git checkout: $upstream_dir"

head="$(git -C "$upstream_dir" rev-parse HEAD 2>/dev/null || true)"
if [[ "$head" == "$upstream_commit" ]] \
  && git -C "$upstream_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  patched_tree_matches "$upstream_dir" \
    || die "checkout contains the FBFM patch plus unrelated local changes"
  echo "fetch_upstream: FBFM overlay already applied: $upstream_dir"
  echo "fetch_upstream: revision $head"
  exit 0
fi

repo_dirty "$upstream_dir" \
  && die "refusing to change a checkout with local changes: $upstream_dir"

if ! git -C "$upstream_dir" cat-file -e "$upstream_commit^{commit}" 2>/dev/null; then
  ((offline == 0)) || die "pinned commit is unavailable locally in --offline mode"
  git -C "$upstream_dir" fetch origin "$upstream_branch" --depth=64 || true
  if ! git -C "$upstream_dir" cat-file -e "$upstream_commit^{commit}" 2>/dev/null; then
    git -C "$upstream_dir" fetch origin "$upstream_commit"
  fi
fi
git -C "$upstream_dir" cat-file -e "$upstream_commit^{commit}" 2>/dev/null \
  || die "pinned commit is unavailable: $upstream_commit"
git -C "$upstream_dir" checkout --detach "$upstream_commit"

git -C "$upstream_dir" apply --check "$patch_file" \
  || die "FBFM patch does not apply to pinned commit $upstream_commit"
git -C "$upstream_dir" apply "$patch_file"
patched_tree_matches "$upstream_dir" \
  || die "checkout differs from the expected patched tree after apply"

echo "fetch_upstream: applied FBFM overlay: $upstream_dir"
echo "fetch_upstream: revision $(git -C "$upstream_dir" rev-parse HEAD)"
