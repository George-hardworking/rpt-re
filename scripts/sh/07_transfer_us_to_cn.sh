#!/usr/bin/env bash
# 07_transfer_us_to_cn: US CNN weights -> China direct / head finetune + backtest + compare.
#
# Usage:
#   ./scripts/sh/07_transfer_us_to_cn.sh all --gpu-ids 0,1,2,3
#   ./scripts/sh/07_transfer_us_to_cn.sh direct --gpu-ids 0,1
#   ./scripts/sh/07_transfer_us_to_cn.sh finetune --gpu-ids 0,1
#   ./scripts/sh/07_transfer_us_to_cn.sh backtest
#   ./scripts/sh/07_transfer_us_to_cn.sh compare
#   ./scripts/sh/07_transfer_us_to_cn.sh direct --image-days 5 --horizon 5 --no-all-seeds --seed 0
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

PY="$ROOT/scripts/py/07_transfer_us_to_cn.py"
if [[ ! -f "$PY" ]]; then
  echo "ERROR: missing Python entry: $PY" >&2
  exit 1
fi

if [[ -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux not installed; run inside tmux or install tmux" >&2
    exit 1
  fi
  mkdir -p "$ROOT/logs/07_transfer_us_cn"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  SESSION="rpt_07_transfer_${STAMP}"
  MAIN_LOG="$ROOT/logs/07_transfer_us_cn/07_transfer_us_to_cn_${STAMP}.log"
  echo "[INFO] Starting tmux session ${SESSION}"
  echo "[INFO] Log file: ${MAIN_LOG}"
  _quoted_args=""
  if (($# > 0)); then
    _quoted_args=$(printf ' %q' "$@")
  fi
  tmux new-session -d -s "$SESSION" \
    env RPT_MAIN_LOG="$MAIN_LOG" RPT_STAMP="$STAMP" \
    bash -lc "cd '$ROOT' && bash '$ROOT/scripts/sh/07_transfer_us_to_cn.sh'${_quoted_args}"
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

LOG_DIR="$ROOT/logs/07_transfer_us_cn"
mkdir -p "$LOG_DIR"
STAMP="${RPT_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAIN_LOG="${RPT_MAIN_LOG:-$LOG_DIR/07_transfer_us_to_cn_${STAMP}.log}"

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
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

RESERVE_GIB="${RPT_RESERVE_GIB:-16}"
if [[ "$*" != *"--reserve-gib"* ]]; then
  set -- "$@" --reserve-gib "$RESERVE_GIB"
fi
if [[ "$*" != *"--batch-size"* ]]; then
  set -- "$@" --batch-size 512
fi

log "============================================================"
log "07_transfer_us_to_cn"
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
  log "ERROR: 07_transfer_us_to_cn failed (exit ${ec}). See ${MAIN_LOG}"
  exit "$ec"
fi

log "Done."
