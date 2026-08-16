"""CLI: paper Table III holding-period split (days 1–5 vs 6–20) for monthly R20.

Uses existing Ix/R20 CNN predictions and step-05 monthly benchmark panels.
Does not retrain or regenerate images. US and CN via --market.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import h1_perf_tables
from backtest.holding_breakdown import attach_r20_subperiod_returns
from backtest.io import (
    load_image_labels,
    load_us_predictions,
    merge_cnn_panel,
    write_h1_excel,
)
from backtest.markets import cn_cnn_spec, us_spec
from config import (
    BACKTEST_BREAKDOWN_ROOT,
    BACKTEST_N_GROUP,
    BACKTEST_WEIGHT_SCHEMES,
    HOLDING_BREAKDOWN_BENCH_COLS,
    HOLDING_BREAKDOWN_CNN_CONFIGS,
    HOLDING_BREAKDOWN_FIRST_DAYS,
    HOLDING_BREAKDOWN_HORIZON,
    HOLDING_BREAKDOWN_WINDOWS,
    HORIZON_DIAGONAL_IMAGE_DAYS,
    MARKET_CN,
    MARKET_US,
    benchmark_signals_dir,
    market_processed_dir,
    market_sample_config,
    sample_freq_for_horizon,
)
from models.dataset import model_run_tag


def log(msg: str) -> None:
    print(msg, flush=True)


def default_test_start(market: str) -> str:
    return market_sample_config(market).test_start


def breakdown_xlsx_path(market: str, window_tag: str, *, direct_signal: bool) -> Path:
    stem = f"direct_{window_tag}" if direct_signal else window_tag
    return BACKTEST_BREAKDOWN_ROOT / market / "monthly" / f"{stem}_h1.xlsx"


def cnn_spec(market: str, image_days: int, horizon: int):
    if market == MARKET_CN:
        return cn_cnn_spec(horizon)
    return us_spec(image_days, horizon)


def load_cnn_breakdown_panel(
    *,
    market: str,
    image_days: int,
    models_root: Path,
    images_root: Path,
    start: str,
) -> tuple[pd.DataFrame, str]:
    horizon = HOLDING_BREAKDOWN_HORIZON
    tag = model_run_tag(image_days, horizon)
    pred = load_us_predictions(models_root, image_days, horizon)
    freq = sample_freq_for_horizon(horizon)
    labels = load_image_labels(
        images_root,
        image_days,
        freq,
        (HOLDING_BREAKDOWN_FIRST_DAYS, horizon),
    )
    spec = cnn_spec(market, image_days, horizon)
    extra = (f"Ret_{HOLDING_BREAKDOWN_FIRST_DAYS}d",)
    panel = merge_cnn_panel(
        pred,
        labels,
        horizon=horizon,
        spec=spec,
        start=start,
        extra_ret_cols=extra,
    )
    panel = attach_r20_subperiod_returns(panel)
    return panel, tag


def load_benchmark_breakdown_panel(
    *,
    market: str,
    signal_col: str,
    images_root: Path,
    start: str,
) -> pd.DataFrame:
    horizon = HOLDING_BREAKDOWN_HORIZON
    path = benchmark_signals_dir(signal_col, market, horizon) / "signals.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run ./scripts/sh/05_benchmark_signals.sh backtest --market {market}"
        )
    panel = pd.read_parquet(path)
    panel["PERMNO"] = panel["PERMNO"].astype("int64")
    panel["Date"] = pd.to_datetime(panel["Date"])
    if signal_col not in panel.columns:
        raise KeyError(f"{path} missing signal column {signal_col}")

    image_days = HORIZON_DIAGONAL_IMAGE_DAYS[horizon]
    freq = sample_freq_for_horizon(horizon)
    labels = load_image_labels(
        images_root,
        image_days,
        freq,
        (HOLDING_BREAKDOWN_FIRST_DAYS, horizon),
    )
    extra = labels[["PERMNO", "Date", f"Ret_{HOLDING_BREAKDOWN_FIRST_DAYS}d"]].copy()
    extra["PERMNO"] = extra["PERMNO"].astype("int64")
    extra["Date"] = pd.to_datetime(extra["Date"])
    r5_col = f"Ret_{HOLDING_BREAKDOWN_FIRST_DAYS}d"
    if r5_col in panel.columns:
        panel = panel.drop(columns=[r5_col])
    panel = panel.merge(extra, on=["PERMNO", "Date"], how="inner")
    panel = panel[panel["Date"] >= pd.Timestamp(start)]
    assert len(panel) > 0, f"empty {signal_col} panel after attaching Ret_5d"
    return attach_r20_subperiod_returns(panel)


def run_window(
    *,
    market: str,
    window_tag: str,
    ret_col: str,
    cnn_panels: list[tuple[str, str, pd.DataFrame, object]],
    bench_panels: list[tuple[str, str, pd.DataFrame, object]],
    ngroup: int,
    direct_signal: bool,
    fresh: bool,
) -> Path:
    out_path = breakdown_xlsx_path(market, window_tag, direct_signal=direct_signal)
    if out_path.is_file() and not fresh:
        log(f"skip breakdown (exists): {out_path}")
        return out_path
    if fresh and out_path.is_file():
        out_path.unlink()

    scheme_rows = {scheme: [] for scheme in BACKTEST_WEIGHT_SCHEMES}
    jobs = cnn_panels + bench_panels
    for row_name, signal_col, panel, spec in jobs:
        window_spec = replace(spec, ret_col=ret_col)
        log(f"{window_tag} {row_name} n={len(panel)} dates={panel['Date'].nunique()}")
        tables = h1_perf_tables(
            panel,
            spec=window_spec,
            signal_cols=[signal_col],
            ngroup=ngroup,
            row_names=[row_name],
            direct_signal=direct_signal,
        )
        for scheme, frame in tables.items():
            scheme_rows[scheme].append(frame)

    combined = {scheme: pd.concat(frames, axis=0) for scheme, frames in scheme_rows.items()}
    path = write_h1_excel(combined, out_path)
    log(f"wrote {path}")
    return path


def run_breakdown(args: argparse.Namespace) -> list[Path]:
    market = args.market
    market_root = market_processed_dir(market)
    images_root = args.images if args.images is not None else market_root / "images"
    models_root = args.models if args.models is not None else market_root / "models"
    start = args.start if args.start is not None else default_test_start(market)
    direct_signal = args.direct_signal

    expected = [
        breakdown_xlsx_path(market, tag, direct_signal=direct_signal)
        for tag, _ in HOLDING_BREAKDOWN_WINDOWS
    ]
    if not args.fresh and all(p.is_file() for p in expected):
        for p in expected:
            log(f"skip breakdown (exists): {p}")
        return expected

    cnn_panels: list[tuple[str, str, pd.DataFrame, object]] = []
    for image_days, horizon in HOLDING_BREAKDOWN_CNN_CONFIGS:
        log(f"load {market.upper()} CNN I{image_days}_R{horizon}")
        panel, tag = load_cnn_breakdown_panel(
            market=market,
            image_days=image_days,
            models_root=models_root,
            images_root=images_root,
            start=start,
        )
        spec = cnn_spec(market, image_days, horizon)
        cnn_panels.append((tag, "p_up", panel, spec))

    bench_panels: list[tuple[str, str, pd.DataFrame, object]] = []
    image_days_diag = HORIZON_DIAGONAL_IMAGE_DAYS[HOLDING_BREAKDOWN_HORIZON]
    spec_bench = cnn_spec(market, image_days_diag, HOLDING_BREAKDOWN_HORIZON)
    for signal_col in HOLDING_BREAKDOWN_BENCH_COLS:
        log(f"load {market.upper()} benchmark {signal_col}")
        panel = load_benchmark_breakdown_panel(
            market=market,
            signal_col=signal_col,
            images_root=images_root,
            start=start,
        )
        bench_panels.append((signal_col, signal_col, panel, spec_bench))

    written: list[Path] = []
    for window_tag, ret_col in HOLDING_BREAKDOWN_WINDOWS:
        written.append(
            run_window(
                market=market,
                window_tag=window_tag,
                ret_col=ret_col,
                cnn_panels=cnn_panels,
                bench_panels=bench_panels,
                ngroup=args.ngroup,
                direct_signal=direct_signal,
                fresh=args.fresh,
            )
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table III monthly R20 holding split: days 1–5 vs 6–20"
    )
    parser.add_argument("--market", choices=(MARKET_US, MARKET_CN), default=MARKET_US)
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--ngroup", type=int, default=BACKTEST_N_GROUP)
    parser.add_argument(
        "--direct-signal",
        action="store_true",
        help="sort deciles on raw signal (paper Table I/III style)",
    )
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_breakdown(args)


if __name__ == "__main__":
    main()
