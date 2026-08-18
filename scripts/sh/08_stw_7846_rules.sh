#!/usr/bin/env bash
set -euo pipefail

# Server-only wrapper for the STW 7,846-rule replication.
#
# TODO on server: set these three paths before running.
: "${STW_OHLC_ROOT:?set STW_OHLC_ROOT to processed/<market>/ohlc_daily}"
: "${STW_UNIVERSE:?set STW_UNIVERSE to a rebalance panel parquet with PERMNO, Date, returns, caps}"
: "${STW_OUTPUT_ROOT:?set STW_OUTPUT_ROOT to the desired output directory}"

MARKET="${MARKET:-us}"
HORIZON="${HORIZON:-5}"
RET_COL="${RET_COL:-Ret_${HORIZON}d}"
RULE_CHUNK_SIZE="${RULE_CHUNK_SIZE:-64}"
STOCK_BATCH_SIZE="${STOCK_BATCH_SIZE:-16}"

args=(
  scripts/py/08_stw_7846_rules.py all
  --market "${MARKET}"
  --horizon "${HORIZON}"
  --ohlc-root "${STW_OHLC_ROOT}"
  --universe "${STW_UNIVERSE}"
  --ret-col "${RET_COL}"
  --output-root "${STW_OUTPUT_ROOT}"
  --rule-chunk-size "${RULE_CHUNK_SIZE}"
  --stock-batch-size "${STOCK_BATCH_SIZE}"
)

# Optional CNN Sharpe values for the red vertical lines in the Figure-8 style plot.
[[ -n "${CNN_SHARPE_EQUAL:-}" ]] && args+=(--cnn-sharpe-equal "${CNN_SHARPE_EQUAL}")
[[ -n "${CNN_SHARPE_FLOAT:-}" ]] && args+=(--cnn-sharpe-float "${CNN_SHARPE_FLOAT}")
[[ -n "${CNN_SHARPE_TOTAL:-}" ]] && args+=(--cnn-sharpe-total "${CNN_SHARPE_TOTAL}")

python "${args[@]}"
