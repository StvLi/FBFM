#!/usr/bin/env bash
# Create isolated model/simulator environments for the FBFM routes.
#
# The script does not download checkpoints or datasets.  It installs source
# dependencies only when --skip-install is not supplied; heavy CUDA packages
# (torch/flash-attn) therefore remain an explicit, visible part of the setup
# log.  Conda (or mamba) is preferred, with a venv fallback for machines that
# already provide the requested Python interpreters.
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
EXTERNAL_ROOT="${FBFM_EXTERNAL_ROOT:-$REPO_ROOT/external}"
ENV_ROOT="${FBFM_ENV_ROOT:-$REPO_ROOT/.venvs}"
ROUTE="all"
BACKEND="${FBFM_ENV_BACKEND:-auto}"
DRY_RUN=0
SKIP_INSTALL=0
SKIP_UPSTREAM=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap/create_envs.sh [options]

Create separate environments for model and simulator processes.  No model
weights, datasets, or simulator assets are downloaded.

Options:
  --route ROUTE          all, lingbot, dreamzero, wan, or sim (default: all)
  --env-root PATH        environment directory (default: ./.venvs)
  --external-root PATH   upstream checkout directory (default: ./external)
  --backend MODE         auto, conda, mamba, or venv (default: auto)
  --skip-install         create environments but do not run pip
  --skip-upstream        install route dependencies but not upstream packages
  --dry-run              print actions without changing the filesystem
  -h, --help             show this help

Environment variables:
  FBFM_*_PYTHON_VERSION  override Python versions (LINGBOT, ROBOTWIN,
                         DREAMZERO, LIBERO, WAN; defaults 3.10, 3.10, 3.11,
                         3.8, 3.10 respectively)
  FBFM_PIP_EXTRA_ARGS    extra arguments (word-split) passed to pip
  FBFM_INSTALL_UPSTREAM  set to 0 (same as --skip-upstream)
  FBFM_ROBOTWIN_MAX_JOBS PyTorch3D/CuRobo build parallelism (default: 4)
  FBFM_WAN_SOURCE_ROOT   patched Wan2.2 checkout (defaults to route vendor,
                         then external/Wan2.2)
  FBFM_WAN_TORCH_REQUIREMENTS
                         requirements file or "existing" (default: audited cu129)
  FBFM_WAN_INSTALL_FLASH_ATTN
                         set to 0 to use the PyTorch SDPA fallback (default: 1)
  FBFM_WAN_MAX_JOBS      FlashAttention build parallelism (default: 4)

Conda environments are prefix-based, so no shell activation is required by
the launchers.  For a prepared machine with existing environments, set
FBFM_ENV_ROOT to their parent and use --skip-install.
EOF
}

die() {
  echo "create_envs: error: $*" >&2
  exit 1
}

log() { echo "create_envs: $*"; }

while (($#)); do
  case "$1" in
    --route)
      (($# >= 2)) || die "--route requires a value"
      ROUTE="$2"; shift 2 ;;
    --route=*) ROUTE="${1#*=}"; shift ;;
    --env-root)
      (($# >= 2)) || die "--env-root requires a path"
      ENV_ROOT="$2"; shift 2 ;;
    --env-root=*) ENV_ROOT="${1#*=}"; shift ;;
    --external-root)
      (($# >= 2)) || die "--external-root requires a path"
      EXTERNAL_ROOT="$2"; shift 2 ;;
    --external-root=*) EXTERNAL_ROOT="${1#*=}"; shift ;;
    --backend)
      (($# >= 2)) || die "--backend requires a value"
      BACKEND="$2"; shift 2 ;;
    --backend=*) BACKEND="${1#*=}"; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-upstream) SKIP_UPSTREAM=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --help)" ;;
  esac
done

case "$ROUTE" in
  all)       ENVIRONMENTS=(lingbot robotwin dreamzero libero wan) ;;
  lingbot)   ENVIRONMENTS=(lingbot robotwin) ;;
  dreamzero) ENVIRONMENTS=(dreamzero libero) ;;
  wan)       ENVIRONMENTS=(wan) ;;
  sim)       ENVIRONMENTS=(robotwin libero) ;;
  *) die "unsupported route '$ROUTE'; choose all, lingbot, dreamzero, wan, or sim" ;;
esac

if [[ "${FBFM_INSTALL_UPSTREAM:-1}" == "0" ]]; then
  SKIP_UPSTREAM=1
fi

case "$BACKEND" in
  auto)
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
      BACKEND=conda
    elif command -v conda >/dev/null 2>&1; then
      BACKEND=conda
    elif command -v mamba >/dev/null 2>&1; then
      BACKEND=mamba
    else
      BACKEND=venv
    fi
    ;;
  conda|mamba|venv) ;;
  *) die "unsupported backend '$BACKEND'; choose auto, conda, mamba, or venv" ;;
esac

if [[ "$BACKEND" == conda ]]; then
  CONDA_CMD="${CONDA_EXE:-conda}"
elif [[ "$BACKEND" == mamba ]]; then
  CONDA_CMD="${MAMBA_EXE:-mamba}"
fi

version_for() {
  case "$1" in
    lingbot)   echo "${FBFM_LINGBOT_PYTHON_VERSION:-3.10}" ;;
    robotwin)  echo "${FBFM_ROBOTWIN_PYTHON_VERSION:-3.10}" ;;
    dreamzero) echo "${FBFM_DREAMZERO_PYTHON_VERSION:-3.11}" ;;
    libero)    echo "${FBFM_LIBERO_PYTHON_VERSION:-3.8}" ;;
    wan)       echo "${FBFM_WAN_PYTHON_VERSION:-3.10}" ;;
    *) die "unknown environment '$1'" ;;
  esac
}

env_path_for() {
  case "$1" in
    lingbot) echo "$ENV_ROOT/fbfm-lingbot-va" ;;
    robotwin) echo "$ENV_ROOT/fbfm-robotwin" ;;
    dreamzero) echo "$ENV_ROOT/fbfm-dreamzero" ;;
    libero) echo "$ENV_ROOT/fbfm-libero" ;;
    wan) echo "$ENV_ROOT/fbfm-wan2.2" ;;
    *) die "unknown environment '$1'" ;;
  esac
}

run() {
  if ((DRY_RUN)); then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

split_extra_args() {
  EXTRA_ARGS=()
  if [[ -n "${FBFM_PIP_EXTRA_ARGS:-}" ]]; then
    IFS=' ' read -r -a EXTRA_ARGS <<<"$FBFM_PIP_EXTRA_ARGS"
  fi
}
split_extra_args

create_env() {
  local key="$1" path="$2" pyver="$3"
  local python="$path/bin/python"
  if ((DRY_RUN)); then
    if [[ "$BACKEND" == conda || "$BACKEND" == mamba ]]; then
      run "$CONDA_CMD" create -y --prefix "$path" "python=$pyver" pip
    else
      local interpreter=""
      case "$key" in
        lingbot) interpreter="${FBFM_LINGBOT_PYTHON:-}" ;;
        robotwin) interpreter="${FBFM_ROBOTWIN_PYTHON:-}" ;;
        dreamzero) interpreter="${FBFM_DREAMZERO_PYTHON:-}" ;;
        libero) interpreter="${FBFM_LIBERO_PYTHON:-}" ;;
        wan) interpreter="${FBFM_WAN_PYTHON:-}" ;;
      esac
      [[ -n "$interpreter" ]] || interpreter="python$pyver"
      run "$interpreter" -m venv "$path"
    fi
    return 0
  fi

  if [[ -x "$python" ]]; then
    log "$key already exists: $path"
    return 0
  fi
  [[ ! -e "$path" ]] || die "environment path exists but has no bin/python: $path"
  mkdir -p "$ENV_ROOT"
  if [[ "$BACKEND" == conda || "$BACKEND" == mamba ]]; then
    "$CONDA_CMD" create -y --prefix "$path" "python=$pyver" pip
  else
    local interpreter=""
    case "$key" in
      lingbot) interpreter="${FBFM_LINGBOT_PYTHON:-}" ;;
      robotwin) interpreter="${FBFM_ROBOTWIN_PYTHON:-}" ;;
      dreamzero) interpreter="${FBFM_DREAMZERO_PYTHON:-}" ;;
      libero) interpreter="${FBFM_LIBERO_PYTHON:-}" ;;
      wan) interpreter="${FBFM_WAN_PYTHON:-}" ;;
    esac
    if [[ -z "$interpreter" ]]; then
      if command -v "python$pyver" >/dev/null 2>&1; then
        interpreter="$(command -v "python$pyver")"
      elif [[ "$pyver" == "3.10" && -n "${FBFM_PYTHON310:-}" ]]; then
        interpreter="$FBFM_PYTHON310"
      elif [[ "$pyver" == "3.11" && -n "${FBFM_PYTHON311:-}" ]]; then
        interpreter="$FBFM_PYTHON311"
      elif [[ "$pyver" == "3.8" && -n "${FBFM_PYTHON38:-}" ]]; then
        interpreter="$FBFM_PYTHON38"
      else
        die "Python $pyver is unavailable for $key; install it or set FBFM_${key^^}_PYTHON"
      fi
    fi
    "$interpreter" -m venv "$path"
  fi
  [[ -x "$python" ]] || die "environment creation did not produce $python"
}

pip_run() {
  local python="$1"; shift
  run "$python" -m pip "$@" "${EXTRA_ARGS[@]}"
}

pip_file() {
  local python="$1" file="$2"; shift 2
  if ((DRY_RUN)); then
    pip_run "$python" install -r "$file" "$@"
  elif [[ -f "$file" ]]; then
    pip_run "$python" install -r "$file" "$@"
  else
    die "dependency file not found: $file"
  fi
}

pip_editable() {
  local python="$1" source="$2"; shift 2
  if ((DRY_RUN)); then
    pip_run "$python" install -e "$source" "$@"
  elif [[ -f "$source/pyproject.toml" || -f "$source/setup.py" ]]; then
    pip_run "$python" install -e "$source" "$@"
  else
    die "Python package metadata not found: $source"
  fi
}

install_for() {
  local key="$1" path="$2" python="$path/bin/python"
  ((SKIP_INSTALL)) && return 0
  if ((DRY_RUN == 0)) && [[ ! -x "$python" ]]; then
    die "cannot install into missing environment $path"
  fi
  pip_run "$python" install --upgrade pip setuptools wheel
  case "$key" in
    lingbot)
      pip_run "$python" install pytest
      pip_file "$python" "$REPO_ROOT/wam/lingbot-va/requirements.txt"
      pip_editable "$python" "$REPO_ROOT/wam/lingbot-va" --no-deps
      # The route directory contains the runnable FBFM integration.  The
      # pinned clean upstream checkout is provenance only; installing it after
      # this package would shadow the FBFM bridge on sys.path.
      ;;
    robotwin)
      if ((SKIP_UPSTREAM == 0)); then
        pip_file "$python" "$EXTERNAL_ROOT/RoboTwin/script/requirements.txt"
        # RoboTwin is executed from its checkout and has no setup.py. Its two
        # compiled planner dependencies are fetched at immutable revisions by
        # fetch_upstreams.sh and built only after the pinned torch install.
        run env "MAX_JOBS=${FBFM_ROBOTWIN_MAX_JOBS:-4}" \
          "$python" -m pip install --no-build-isolation -e \
          "$EXTERNAL_ROOT/pytorch3d" "${EXTRA_ARGS[@]}"
        run env "MAX_JOBS=${FBFM_ROBOTWIN_MAX_JOBS:-4}" \
          "$python" -m pip install --no-build-isolation -e \
          "$EXTERNAL_ROOT/curobo" "${EXTRA_ARGS[@]}"
        run "$python" "$REPO_ROOT/scripts/bootstrap/apply_robotwin_compat.py"
        run "$python" -c \
          'import importlib.util as u; assert all(u.find_spec(x) for x in ("pytorch3d", "curobo", "sapien", "mplib")); print("RoboTwin runtime imports OK")'
      fi
      ;;
    dreamzero)
      pip_run "$python" install pytest
      if ((SKIP_UPSTREAM == 0)); then
        pip_editable "$python" "$EXTERNAL_ROOT/dreamzero"
        pip_editable "$python" "$EXTERNAL_ROOT/RLinf" --no-deps
      fi
      pip_editable "$python" "$REPO_ROOT/wam/dreamzero-libero" --no-deps
      ;;
    libero)
      if ((SKIP_UPSTREAM == 0)); then
        pip_file "$python" "$EXTERNAL_ROOT/LIBERO/requirements.txt"
        pip_editable "$python" "$EXTERNAL_ROOT/LIBERO" --no-deps
      fi
      ;;
    wan)
      wan_environment="$REPO_ROOT/wam/wan2.2/environment"
      wan_source="${FBFM_WAN_SOURCE_ROOT:-}"
      if [[ -z "$wan_source" && -f "$REPO_ROOT/wam/wan2.2/vendor/Wan2.2/pyproject.toml" ]]; then
        wan_source="$REPO_ROOT/wam/wan2.2/vendor/Wan2.2"
      elif [[ -z "$wan_source" ]]; then
        wan_source="$EXTERNAL_ROOT/Wan2.2"
      fi

      # Install in phases. FlashAttention's build imports torch, so the
      # upstream all-in-one requirements.txt is not reproducible in a fresh env.
      wan_torch_requirements="${FBFM_WAN_TORCH_REQUIREMENTS:-$wan_environment/requirements-torch-cu129.txt}"
      if [[ "$wan_torch_requirements" == existing ]]; then
        run "$python" -c 'import torch, torchvision, torchaudio; print(torch.__version__)'
      else
        pip_file "$python" "$wan_torch_requirements"
      fi
      pip_file "$python" "$wan_environment/requirements-runtime.txt"

      if [[ "${FBFM_WAN_INSTALL_FLASH_ATTN:-1}" == 0 ]]; then
        log "wan: skipping optional FlashAttention; PyTorch SDPA remains available"
      else
        run env "MAX_JOBS=${FBFM_WAN_MAX_JOBS:-4}" \
          "$python" -m pip install --no-build-isolation \
          -r "$wan_environment/requirements-flash-attn.txt" "${EXTRA_ARGS[@]}"
      fi

      if ((SKIP_UPSTREAM == 0)); then
        pip_editable "$python" "$wan_source" --no-deps
      fi
      ;;
  esac
}

log "backend=$BACKEND env_root=$ENV_ROOT external_root=$EXTERNAL_ROOT"
for key in "${ENVIRONMENTS[@]}"; do
  path="$(env_path_for "$key")"
  create_env "$key" "$path" "$(version_for "$key")"
  install_for "$key" "$path"
done

if ((DRY_RUN)); then
  log "dry-run complete; no environments or packages were changed"
else
  log "environment setup complete"
  for key in "${ENVIRONMENTS[@]}"; do
    path="$(env_path_for "$key")"
    echo "  $key: $path/bin/python"
  done
  echo "Use the printed Python executables directly; activation is optional."
fi
