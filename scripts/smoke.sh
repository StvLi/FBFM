#!/usr/bin/env bash
# Fast CPU/import smoke tests for the public FBFM checkout.  These checks are
# intentionally bounded: they never launch a full GPU episode or download a
# checkpoint.  A Wan VAE check can be requested explicitly with --model.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
ROUTE="all"
MODEL="${FBFM_SMOKE_MODEL:-}"
FAILURES=0

usage() {
  cat <<'EOF'
Usage: scripts/smoke.sh [ROUTE] [options]

ROUTE may be shared, lingbot, dreamzero, wan, or all (default: all).

Options:
  --route ROUTE          same as the positional route
  --model PATH           run the optional Wan streaming-VAE check on PATH
  --help                 show this help

Python overrides (otherwise .venvs/fbfm-*/bin/python, then python3):
  FBFM_SHARED_PYTHON, FBFM_LINGBOT_PYTHON, FBFM_DREAMZERO_PYTHON,
  FBFM_WAN_PYTHON, FBFM_ENV_ROOT
EOF
}

die() { echo "smoke: error: $*" >&2; exit 2; }

if (($#)) && [[ "$1" != -* ]]; then
  ROUTE="$1"
  shift
fi
while (($#)); do
  case "$1" in
    --route)
      (($# >= 2)) || die "--route requires a value"
      ROUTE="$2"; shift 2 ;;
    --route=*) ROUTE="${1#*=}"; shift ;;
    --model)
      (($# >= 2)) || die "--model requires a checkpoint path"
      MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
done

case "$ROUTE" in
  all)       STEPS=(shared lingbot dreamzero wan) ;;
  shared|lingbot|dreamzero|wan) STEPS=("$ROUTE") ;;
  *) die "unsupported route '$ROUTE'; choose shared, lingbot, dreamzero, wan, or all" ;;
esac

env_root="${FBFM_ENV_ROOT:-$REPO_ROOT/.venvs}"
python_for() {
  local key="$1" override="" candidate
  case "$key" in
    shared) override="${FBFM_SHARED_PYTHON:-}"; candidate="$env_root/fbfm-dreamzero/bin/python" ;;
    lingbot) override="${FBFM_LINGBOT_PYTHON:-}"; candidate="$env_root/fbfm-lingbot-va/bin/python" ;;
    dreamzero) override="${FBFM_DREAMZERO_PYTHON:-}"; candidate="$env_root/fbfm-dreamzero/bin/python" ;;
    wan) override="${FBFM_WAN_PYTHON:-}"; candidate="$env_root/fbfm-wan2.2/bin/python" ;;
    *) die "unknown smoke interpreter '$key'" ;;
  esac
  if [[ -n "$override" ]]; then
    echo "$override"
  elif [[ -x "$candidate" ]]; then
    echo "$candidate"
  else
    command -v python3 || echo python3
  fi
}

run_step() {
  local label="$1"; shift
  echo "smoke: [$label]"
  if "$@"; then
    echo "smoke: [$label] PASS"
  else
    status=$?
    echo "smoke: [$label] FAIL (exit $status)" >&2
    FAILURES=$((FAILURES + 1))
  fi
}

run_shared() {
  local py; py="$(python_for shared)"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -c 'from fbfm.configs.types import RTCAttentionSchedule; from fbfm.libero_observation import quaternion_xyzw_to_axis_angle; assert RTCAttentionSchedule.EXP.value == "EXP"; print("shared fbfm import OK")'
}

run_lingbot() {
  local py; py="$(python_for lingbot)"
  [[ -d "$REPO_ROOT/wam/lingbot-va/tests" ]] || { echo "smoke: LingBot route is absent; skipping" >&2; return 0; }
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH="$REPO_ROOT:$REPO_ROOT/wam/lingbot-va${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m pytest -p no:cacheprovider -q \
      "$REPO_ROOT/wam/lingbot-va/tests/test_fbfm_bridge.py" \
      "$REPO_ROOT/wam/lingbot-va/tests/test_async_transport.py"
}

run_dreamzero() {
  local py; py="$(python_for dreamzero)"
  [[ -d "$REPO_ROOT/wam/dreamzero-libero/tests" ]] || { echo "smoke: DreamZero route is absent; skipping" >&2; return 0; }
  # The root-module contract test intentionally has two deployment-layout
  # assertions that only apply to the standalone A6000 checkout.  Route tests
  # are all CPU-only and do not import the model checkpoint.
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH="$REPO_ROOT/wam/dreamzero-libero/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m pytest -p no:cacheprovider -q \
      --ignore="$REPO_ROOT/wam/dreamzero-libero/tests/test_root_fbfm_modules.py" \
      "$REPO_ROOT/wam/dreamzero-libero/tests"
}

run_wan() {
  local py; py="$(python_for wan)"
  local root="$REPO_ROOT/wam/wan2.2"
  if [[ ! -d "$root" ]]; then
    echo "smoke: Wan2.2 overlay is absent; run the submission assembly first (skipping)" >&2
    return 0
  fi
  [[ -f "$root/tests/test_fbfm_state_feedback.py" ]] || {
    echo "smoke: Wan2.2 FBFM tests are absent; skipping" >&2
    return 0
  }
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    PYTHONPATH="$root:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m pytest -p no:cacheprovider -q "$root/tests/test_fbfm_state_feedback.py"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" "$root/generate_fbfm.py" --help >/dev/null
  if [[ -n "$MODEL" ]]; then
    [[ -f "$root/scripts/validate_fbfm_vae.py" ]] || die "Wan VAE validator is missing from the overlay"
    [[ -d "$MODEL" ]] || die "Wan VAE model directory does not exist: $MODEL"
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$root:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$py" "$root/scripts/validate_fbfm_vae.py" --ckpt-dir "$MODEL"
  fi
}

for step in "${STEPS[@]}"; do
  case "$step" in
    shared) run_step shared run_shared ;;
    lingbot) run_step lingbot run_lingbot ;;
    dreamzero) run_step dreamzero run_dreamzero ;;
    wan) run_step wan run_wan ;;
  esac
done

if ((FAILURES)); then
  echo "smoke: $FAILURES step(s) failed" >&2
  exit 1
fi
echo "smoke: all requested checks passed"
