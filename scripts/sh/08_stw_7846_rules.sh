#!/usr/bin/env bash
# 08_stw_7846_rules: STW 7,846 technical rules vs CNN Figure-8 Sharpe distributions.
#
# Usage:
#   ./scripts/sh/08_stw_7846_rules.sh
#   ./scripts/sh/08_stw_7846_rules.sh all --market us
#   ./scripts/sh/08_stw_7846_rules.sh all --market us --horizons 5
#   ./scripts/sh/08_stw_7846_rules.sh signals --market us --horizon 20
#   ./scripts/sh/08_stw_7846_rules.sh all --market cn   # same CLI; run when CN data ready
#
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$_SCRIPT_DIR"
while [[ ! -f "$ROOT/pyproject.toml" && "$ROOT" != "/" ]]; do
  ROOT="$(dirname "$ROOT")"
done
if [[ ! -f "$ROOT/pyproject.toml" ]]; then
  echo "ERROR: could not locate repo root (pyproject.toml) from $_SCRIPT_DIR" >&2
  exit 1
fi

PY="$ROOT/scripts/py/08_stw_7846_rules.py"
if [[ ! -f "$PY" ]]; then
  echo "ERROR: missing Python entry: $PY" >&2
  exit 1
fi

if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed; run inside tmux or install tmux" >&2
    exit 1
  fi
  mkdir -p "$ROOT/logs/08_stw_7846_rules"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SESSION="rpt_08_stw_${STAMP}"
  MAIN_LOG="$ROOT/logs/08_stw_7846_rules/08_stw_7846_rules_${STAMP}.log"
  echo "[INFO] Starting tmux session ${SESSION}"
  echo "[INFO] Log file: ${MAIN_LOG}"
  _quoted_args=""
  if (($# > 0)); then
    _quoted_args=$(printf ' %q' "$@")
  else
    _quoted_args=" all --market us"
  fi
  tmux new-session -d -s "$SESSION" \
    env RPT_MAIN_LOG="$MAIN_LOG" RPT_STAMP="$STAMP" \
    bash -lc "cd '$ROOT' && bash '$ROOT/scripts/sh/08_stw_7846_rules.sh'${_quoted_args}"
  echo "[INFO] Attach:  tmux attach -t ${SESSION}"
  echo "[INFO] Tail log: tail -f ${MAIN_LOG}"
  exit 0
fi

_conda_sh=""
if command -v conda >/dev/null 2>&1; then
  _conda_sh="$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  _conda_sh="${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  _conda_sh="${HOME}/anaconda3/etc/profile.d/conda.sh"
fi
if [[ -z "${_conda_sh}" || ! -f "${_conda_sh}" ]]; then
  echo "ERROR: conda not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${_conda_sh}"
conda activate 5020_env

cd "$ROOT"

LOG_DIR="$ROOT/logs/08_stw_7846_rules"
mkdir -p "$LOG_DIR"
STAMP="${RPT_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAIN_LOG="${RPT_MAIN_LOG:-$LOG_DIR/08_stw_7846_rules_${STAMP}.log}"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$MAIN_LOG"; }

ts_pipe() {
  while IFS= read -r line; do
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line" | tee -a "$MAIN_LOG"
  done
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RESERVE_GIB="${RPT_RESERVE_GIB:-16}"
if [[ "$*" != *"--reserve-gib"* ]]; then
  set -- "$@" --reserve-gib "$RESERVE_GIB"
fi

if (($# == 0)); then
  set -- all --market us
fi

log "============================================================"
log "08_stw_7846_rules"
log "Conda: ${CONDA_DEFAULT_ENV:-unknown}"
log "Python: $PY"
log "Args:  $*"
log "Log:   ${MAIN_LOG}"
log "============================================================"
log "starting python"

set +e
python "$PY" "$@" 2>&1 | ts_pipe
ec="${PIPESTATUS[0]}"
set -e

if [[ "$ec" -ne 0 ]]; then
  log "ERROR: 08_stw_7846_rules failed (exit ${ec}). See ${MAIN_LOG}"
  exit "$ec"
fi

log "Done."
