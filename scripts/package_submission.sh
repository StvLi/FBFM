#!/usr/bin/env bash
set -euo pipefail

# Create the exact source archive submitted to AAAI.  Only committed files are
# archived; this prevents local checkpoints, simulator outputs and raw sensor
# bags from leaking into the release.
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
branch=${FBFM_SUBMISSION_BRANCH:-submission}
output=${1:-"$repo/../FBFM-submission.tar.gz"}
prefix=${FBFM_ARCHIVE_PREFIX:-FBFM-submission/}

[[ -d "$repo/.git" ]] || { echo "not a git worktree: $repo" >&2; exit 2; }
current=$(git -C "$repo" branch --show-current)
[[ "$current" == "$branch" ]] || {
  echo "package from branch '$branch' (currently '$current')" >&2
  exit 2
}
if [[ -n $(git -C "$repo" status --porcelain) ]]; then
  echo "working tree is dirty; commit the intended release before packaging" >&2
  git -C "$repo" status --short >&2
  exit 2
fi

mkdir -p "$(dirname -- "$output")"
git -C "$repo" archive --format=tar.gz --prefix="$prefix" \
  --output="$output" "$branch"

# Defense-in-depth checks on the generated archive.  These are release policy,
# not substitutes for reviewing the source and third-party licenses.
forbidden_re='(^|/)([^/]*\.(db3|bag|mcap|pt|pth|ckpt|safetensors|onnx|engine)|checkpoints?(/|$)|external(/|$)|\.venvs?(/|$)|\.cache/|cache/|caches/|__pycache__(/|$)|[^/]*\.pyc$)'
if tar -tzf "$output" | grep -Eiq "$forbidden_re"; then
  echo "archive contains a forbidden generated/sensitive path" >&2
  tar -tzf "$output" | grep -Ei "$forbidden_re" >&2 || true
  rm -f -- "$output"
  exit 3
fi

sha256sum "$output" | tee "$output.sha256"
du -h "$output"
echo "wrote $output"
