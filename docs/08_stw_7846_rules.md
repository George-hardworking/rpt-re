# 08 STW 7,846 Technical Trading Rules

This pipeline implements the Figure-8 style benchmark from Jiang, Kelly, and Xiu:
generate the Sullivan, Timmermann, and White (1999) 7,846 technical trading rule
universe, compute stock-level rebalance-date signals, then estimate the distribution
of high-minus-low decile Sharpe ratios.

## Files

- `src/data/stw_rules.py`: deterministic 7,846-rule manifest.
- `src/data/stw_signals.py`: per-stock signal engine with rolling-stat caches.
- `src/backtest/stw_fast.py`: fast NumPy H-L Sharpe computation. The ranking
  path matches the existing repository backtest convention: per-date z-score,
  +/-3 winsorization, second z-score, then `rank(method="first")` deciles.
- `scripts/py/08_stw_7846_rules.py`: CLI entry point.
- `scripts/sh/08_stw_7846_rules.sh`: server wrapper with path placeholders.

## Required Server Inputs

The code deliberately does not hard-code data locations. Pass these on the server:

- `--ohlc-root`: PERMNO-partitioned daily OHLC parquet, with at least
  `DlyCalDt`, `DlyClose`, and `DlyVol`.
- `--universe`: rebalance-date parquet panel with at least `PERMNO`, `Date`,
  a forward return column such as `Ret_5d`, `Ret_20d`, or `Ret_60d`, and optional
  cap columns `FloatCap`, `TotalCap`, or `MarketCap`.
- `--output-root`: destination for manifest, rule chunks, and Sharpe summaries.

Example:

```bash
python scripts/py/08_stw_7846_rules.py all \
  --market us \
  --horizon 5 \
  --ohlc-root /TODO/processed/us/ohlc_daily \
  --universe /TODO/rebalance_panel_weekly.parquet \
  --ret-col Ret_5d \
  --output-root /TODO/outputs/08_stw_7846_rules \
  --cnn-sharpe-equal 7.20 \
  --cnn-sharpe-total 1.70
```

The `--cnn-sharpe-*` values are optional inputs used only for the red vertical
line in the distribution plot. Fill them with the CNN H-L Sharpe ratios from
the corresponding horizon and weight scheme.

## Outputs

For each market/horizon:

- `stw_7846_manifest.csv`: all rules and parameters.
- `rule_chunks/chunk_XXXX/part_YYYYY.parquet`: rebalance-date rule signals.
- `sharpe_chunks/summary_chunk_XXXX.parquet`: per-chunk H-L Sharpe summaries.
- `stw_7846_sharpes.parquet` and `.csv`: combined 7,846-rule Sharpe distribution.
- `figures/figure8_stw_sharpe_distribution.png`: STW Sharpe histogram with
  optional red CNN Sharpe line.
- `figures/figure8_stw_sharpe_distribution_equal_weight.png`: equal-weight
  standalone histogram.
- `figures/figure8_stw_sharpe_distribution_float_value_weight.png`: float-cap
  value-weight standalone histogram, if `FloatCap` is available.
- `figures/figure8_stw_sharpe_distribution_total_value_weight.png`: total-cap
  value-weight standalone histogram, if `TotalCap` is available.

The output tree is:

```text
<output-root>/
  <market>/
    <weekly|monthly|quarterly>/
      stw_7846_manifest.csv
      rule_chunks/
        chunk_0000/
          part_00000.parquet
          part_00001.parquet
          ...
        chunk_0001/
          part_00000.parquet
          ...
      sharpe_chunks/
        summary_chunk_0000.parquet
        summary_chunk_0001.parquet
        ...
      stw_7846_sharpes.parquet
      stw_7846_sharpes.csv
      figures/
        figure8_stw_sharpe_distribution.png
        figure8_stw_sharpe_distribution_equal_weight.png
        figure8_stw_sharpe_distribution_float_value_weight.png
        figure8_stw_sharpe_distribution_total_value_weight.png
```

## Calibration Note

The family counts match the public STW universe count:
497 FR, 2,049 MA, 1,220 SR, 2,040 CB, and 2,040 OBV. Without the original
Scaillet code, exact edge-case behavior should be calibrated later, especially
the small support/resistance stop-unwind block, tie handling in discrete signals,
and fixed-holding-period behavior.
