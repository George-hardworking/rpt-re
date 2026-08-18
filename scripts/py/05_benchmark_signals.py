"""CLI: build paper trend benchmarks and H1 backtest vs CNN universe."""

from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool, Value
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RESERVE_GIB = 16.0
MEM_PER_WORKER_GIB = 0.8
PROGRESS_EVERY = 500

from analysis.cnn_characteristics_tables import run_all_tables
from backtest.engine import h1_perf_tables
from backtest.eval_horizon import align_panel_eval_horizon, backtest_output_stem, parse_eval_horizons
from backtest.io import load_image_labels, write_h1_excel
from backtest.markets import cn_cnn_spec, us_spec
from config import (
    BACKTEST_N_GROUP,
    BENCHMARK_SIGNAL_COLS,
    EVAL_HORIZONS,
    HORIZON_DIAGONAL_IMAGE_DAYS,
    MARKET_CN,
    MARKET_US,
    WINDOW_DAYS,
    benchmark_output_dir,
    benchmark_signals_dir,
    characteristics_month_end_path,
    market_processed_dir,
    market_sample_config,
    market_vw_returns_path,
    sample_freq_for_horizon,
)
from data.market_returns import read_market_vw_returns, write_market_vw_returns
from data.parquet_io import permno_list, read_stock
from data.trend_signals import (
    STOCK_SIGNAL_COLS,
    append_liquidity_characteristics,
    compute_hzz_trend_scores,
    join_signals_at_dates,
    liquidity_partition_complete,
    read_stock_trend_signals,
    trend_signal_partition_complete,
    trend_signals_daily_root,
    write_stock_trend_signals,
)
from utils.workers import resolve_workers

_PROGRESS = None
_TOTAL_STOCKS = None
_SKIP_STOCKS = None


def log(msg: str) -> None:
    print(msg, flush=True)


def chunk_permnos(seq: np.ndarray, n: int) -> list[np.ndarray]:
    k, m = divmod(len(seq), n)
    return [seq[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def _init_progress(progress, total_stocks: int, skip_stocks: int) -> None:
    global _PROGRESS, _TOTAL_STOCKS, _SKIP_STOCKS
    _PROGRESS = progress
    _TOTAL_STOCKS = total_stocks
    _SKIP_STOCKS = skip_stocks


def process_permno_chunk(task: tuple) -> tuple[int, int]:
    permno_chunk, ohlc_path, output_root, worker_id = task
    built = 0
    skipped = 0
    for i, permno in enumerate(permno_chunk):
        permno = int(permno)
        if trend_signal_partition_complete(output_root, permno):
            skipped += 1
        else:
            stock_df = read_stock(ohlc_path, permno)
            write_stock_trend_signals(stock_df, permno, output_root)
            built += 1
        with _PROGRESS.get_lock():
            _PROGRESS.value += 1
            done = _PROGRESS.value
        if done % PROGRESS_EVERY == 0 or done == _TOTAL_STOCKS:
            pct = 100.0 * done / _TOTAL_STOCKS
            log(
                f"progress {pct:5.1f}%  stocks {done}/{_TOTAL_STOCKS}  "
                f"skip={_SKIP_STOCKS}  worker={worker_id} "
                f"chunk={i + 1}/{len(permno_chunk)}"
            )
    return built, skipped


def process_liquidity_chunk(task: tuple) -> tuple[int, int]:
    permno_chunk, ohlc_path, output_root, mkt_ret, worker_id = task
    built = 0
    skipped = 0
    for i, permno in enumerate(permno_chunk):
        permno = int(permno)
        if liquidity_partition_complete(output_root, permno):
            skipped += 1
        else:
            stock_df = read_stock(ohlc_path, permno)
            append_liquidity_characteristics(stock_df, permno, output_root, mkt_ret)
            built += 1
        with _PROGRESS.get_lock():
            _PROGRESS.value += 1
            done = _PROGRESS.value
        if done % PROGRESS_EVERY == 0 or done == _TOTAL_STOCKS:
            pct = 100.0 * done / _TOTAL_STOCKS
            log(
                f"liquidity progress {pct:5.1f}%  stocks {done}/{_TOTAL_STOCKS}  "
                f"skip={_SKIP_STOCKS}  worker={worker_id} "
                f"chunk={i + 1}/{len(permno_chunk)}"
            )
    return built, skipped


def build_daily_signals(
    *,
    market: str,
    ohlc_path: Path,
    output_root: Path,
    permno_limit: int | None,
    n_workers: int | None,
    reserve_gib: float,
    fresh: bool,
) -> Path:
    if not ohlc_path.exists():
        raise FileNotFoundError(f"OHLC parquet not found: {ohlc_path}; run 01_prepare_data ohlc first")

    if fresh and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    permnos = permno_list(ohlc_path)
    if permno_limit is not None:
        permnos = permnos[:permno_limit]

    skip_stocks = sum(
        1 for p in permnos if trend_signal_partition_complete(output_root, int(p))
    )
    pending = len(permnos) - skip_stocks
    log(
        f"{market} trend signals resume: skip={skip_stocks} "
        f"pending={pending} total={len(permnos)}"
    )

    n_workers, diag = resolve_workers(
        pending if pending > 0 else 1,
        override=n_workers,
        reserve_gib=reserve_gib,
        mem_per_worker_gib=MEM_PER_WORKER_GIB,
    )
    log(f"workers={n_workers} diag={diag}")

    if pending == 0:
        log(f"skip daily signals (complete): {output_root}")
        return output_root

    progress = Value("i", skip_stocks)
    total_stocks = len(permnos)
    chunks = chunk_permnos(permnos, n_workers)
    tasks = [(chunk, ohlc_path, output_root, wid) for wid, chunk in enumerate(chunks)]

    if n_workers == 1:
        _init_progress(progress, total_stocks, skip_stocks)
        built, skipped = process_permno_chunk(tasks[0])
    else:
        with Pool(
            n_workers,
            initializer=_init_progress,
            initargs=(progress, total_stocks, skip_stocks),
        ) as pool:
            chunk_stats = pool.map(process_permno_chunk, tasks)
        built = sum(b for b, _ in chunk_stats)
        skipped = sum(s for _, s in chunk_stats)

    log(f"done daily signals: built={built} skip={skipped} -> {output_root}")
    return output_root


def build_liquidity_characteristics(
    *,
    market: str,
    ohlc_path: Path,
    output_root: Path,
    permno_limit: int | None,
    n_workers: int | None,
    reserve_gib: float,
) -> Path:
    mkt_path = market_vw_returns_path(market)
    if not mkt_path.is_file():
        log(f"building market VW returns -> {mkt_path}")
        write_market_vw_returns(ohlc_path, market, out_path=mkt_path)
    mkt_ret = read_market_vw_returns(market, path=mkt_path)

    permnos = permno_list(ohlc_path)
    if permno_limit is not None:
        permnos = permnos[:permno_limit]

    skip_stocks = sum(
        1 for p in permnos if liquidity_partition_complete(output_root, int(p))
    )
    pending = len(permnos) - skip_stocks
    log(
        f"{market} liquidity chars resume: skip={skip_stocks} "
        f"pending={pending} total={len(permnos)}"
    )
    if pending == 0:
        log(f"skip liquidity characteristics (complete): {output_root}")
        return output_root

    n_workers, diag = resolve_workers(
        pending,
        override=n_workers,
        reserve_gib=reserve_gib,
        mem_per_worker_gib=MEM_PER_WORKER_GIB,
    )
    log(f"liquidity workers={n_workers} diag={diag}")

    progress = Value("i", skip_stocks)
    total_stocks = len(permnos)
    chunks = chunk_permnos(permnos, n_workers)
    tasks = [
        (chunk, ohlc_path, output_root, mkt_ret, wid)
        for wid, chunk in enumerate(chunks)
    ]

    if n_workers == 1:
        _init_progress(progress, total_stocks, skip_stocks)
        built, skipped = process_liquidity_chunk(tasks[0])
    else:
        with Pool(
            n_workers,
            initializer=_init_progress,
            initargs=(progress, total_stocks, skip_stocks),
        ) as pool:
            chunk_stats = pool.map(process_liquidity_chunk, tasks)
        built = sum(b for b, _ in chunk_stats)
        skipped = sum(s for _, s in chunk_stats)

    log(f"done liquidity characteristics: built={built} skip={skipped} -> {output_root}")
    return output_root


def load_diagonal_universe(
    images_root: Path,
    horizon: int,
) -> pd.DataFrame:
    image_days = HORIZON_DIAGONAL_IMAGE_DAYS[horizon]
    freq = sample_freq_for_horizon(horizon)
    labels = load_image_labels(images_root, image_days, freq, (horizon,))
    ret_col = f"Ret_{horizon}d"
    cols = ["PERMNO", "Date", ret_col]
    if "FloatCap" in labels.columns and "TotalCap" in labels.columns:
        cols.extend(["FloatCap", "TotalCap"])
    else:
        cols.append("MarketCap")
    return labels[cols].copy()


def _available_permnos(signals_root: Path) -> set[int]:
    return {int(p.name.split("=", 1)[1]) for p in Path(signals_root).glob("PERMNO=*")}


def _universe_signal_rows(
    universe: pd.DataFrame,
    signals_root: Path,
) -> pd.DataFrame:
    avail = _available_permnos(signals_root)
    universe = universe[universe["PERMNO"].isin(avail)].copy()
    assert len(universe) > 0, "no universe rows with built trend signals"
    rows: list[pd.DataFrame] = []
    for permno, grp in universe.groupby("PERMNO", sort=False):
        permno = int(permno)
        daily = read_stock_trend_signals(signals_root, permno)
        as_ofs = pd.DatetimeIndex(grp["Date"].unique())
        sig = join_signals_at_dates(daily, as_ofs, cols=STOCK_SIGNAL_COLS)
        sig["PERMNO"] = permno
        merged = grp.merge(sig, on=["PERMNO", "Date"], how="inner")
        rows.append(merged)
    if not rows:
        raise ValueError("empty panel after joining daily trend signals")
    return pd.concat(rows, ignore_index=True)


def build_freq_signal_panels(
    *,
    market: str,
    images_root: Path,
    signals_root: Path,
    start: str,
    horizons: tuple[int, ...],
) -> dict[int, pd.DataFrame]:
    universes: dict[int, pd.DataFrame] = {}
    permno_sets: list[set[int]] = []
    avail = _available_permnos(signals_root)
    for horizon in horizons:
        universe = load_diagonal_universe(images_root, horizon)
        universe = universe[universe["Date"] >= pd.Timestamp(start)].copy()
        universe = universe[universe["PERMNO"].isin(avail)].copy()
        assert len(universe) > 0, f"empty universe market={market} horizon={horizon}"
        universes[horizon] = universe
        permno_sets.append(set(universe["PERMNO"].astype(int).unique()))

    all_permnos = sorted(set().union(*permno_sets))
    chunks: dict[int, list[pd.DataFrame]] = {h: [] for h in horizons}

    for permno in all_permnos:
        daily = read_stock_trend_signals(signals_root, permno)
        for horizon in horizons:
            grp = universes[horizon]
            sub = grp[grp["PERMNO"] == permno]
            if sub.empty:
                continue
            as_ofs = pd.DatetimeIndex(sub["Date"].unique())
            sig = join_signals_at_dates(daily, as_ofs, cols=STOCK_SIGNAL_COLS)
            sig["PERMNO"] = permno
            chunks[horizon].append(sub.merge(sig, on=["PERMNO", "Date"], how="inner"))

    panels: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        if not chunks[horizon]:
            raise ValueError(f"empty panel market={market} horizon={horizon}")
        panel = pd.concat(chunks[horizon], ignore_index=True)
        ret_col = f"Ret_{horizon}d"
        hzz = compute_hzz_trend_scores(panel, ret_col=ret_col)
        panel["TREND_HZZ"] = hzz.reindex(panel.index).to_numpy(dtype=np.float32)
        panels[horizon] = panel
        log(
            f"{market} horizon={horizon} panel n={len(panel)} "
            f"dates={panel['Date'].nunique()}"
        )
    return panels


def write_signal_parquets(
    panels: dict[int, pd.DataFrame],
    *,
    market: str,
    signal_col: str,
) -> list[Path]:
    written: list[Path] = []
    for horizon, panel in panels.items():
        out_dir = benchmark_signals_dir(signal_col, market, horizon)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "signals.parquet"
        keep = ["PERMNO", "Date", signal_col]
        ret_col = f"Ret_{horizon}d"
        for col in (ret_col, "FloatCap", "TotalCap", "MarketCap"):
            if col in panel.columns and col not in keep:
                keep.append(col)
        panel[keep].to_parquet(path, index=False)
        written.append(path)
        log(f"wrote {path}")
    return written


def backtest_signal(
    panels: dict[int, pd.DataFrame],
    *,
    market: str,
    signal_col: str,
    ngroup: int,
    eval_horizons: tuple[int, ...],
    fresh: bool,
) -> list[Path]:
    written: list[Path] = []
    for horizon, panel in panels.items():
        if market == MARKET_CN:
            spec = cn_cnn_spec(horizon)
        else:
            spec = us_spec(HORIZON_DIAGONAL_IMAGE_DAYS[horizon], horizon)

        for eval_h in eval_horizons:
            out_path = benchmark_output_dir(signal_col, market, horizon) / f"h{eval_h}.xlsx"
            if out_path.is_file() and not fresh:
                log(f"skip backtest (exists): {out_path}")
                written.append(out_path)
                continue
            if fresh and out_path.is_file():
                out_path.unlink()

            aligned = align_panel_eval_horizon(panel, spec, eval_h)
            tables = h1_perf_tables(
                aligned,
                spec=spec,
                signal_cols=[signal_col],
                ngroup=ngroup,
                row_names=[signal_col],
            )
            path = write_h1_excel(tables, out_path)
            log(f"wrote {path}")
            written.append(path)
    return written


def run_signals(args: argparse.Namespace) -> Path:
    market_root = market_processed_dir(args.market)
    ohlc_path = args.ohlc if args.ohlc is not None else market_root / "ohlc_daily"
    output_root = (
        args.signals_root
        if args.signals_root is not None
        else trend_signals_daily_root(args.market)
    )
    build_daily_signals(
        market=args.market,
        ohlc_path=ohlc_path,
        output_root=output_root,
        permno_limit=args.permno_limit,
        n_workers=args.workers,
        reserve_gib=args.reserve_gib,
        fresh=args.fresh,
    )
    return build_liquidity_characteristics(
        market=args.market,
        ohlc_path=ohlc_path,
        output_root=output_root,
        permno_limit=args.permno_limit,
        n_workers=args.workers,
        reserve_gib=args.reserve_gib,
    )


def run_tables(args: argparse.Namespace) -> list[Path]:
    market_root = market_processed_dir(args.market)
    ohlc_path = args.ohlc if args.ohlc is not None else market_root / "ohlc_daily"
    signals_root = (
        args.signals_root
        if args.signals_root is not None
        else trend_signals_daily_root(args.market)
    )
    models_root = args.models if args.models is not None else market_root / "models"

    panel_path = characteristics_month_end_path(args.market)
    if args.fresh and panel_path.is_file():
        panel_path.unlink()

    return run_all_tables(
        market=args.market,
        ohlc_path=ohlc_path,
        signals_root=signals_root,
        models_root=models_root,
        fresh=args.fresh,
        n_workers=args.workers,
        reserve_gib=args.reserve_gib,
    )


def run_backtest(args: argparse.Namespace) -> list[Path]:
    market_root = market_processed_dir(args.market)
    images_root = args.images if args.images is not None else market_root / "images"
    signals_root = (
        args.signals_root
        if args.signals_root is not None
        else trend_signals_daily_root(args.market)
    )
    start = args.start if args.start is not None else market_sample_config(args.market).test_start
    horizons = tuple(args.horizons) if args.horizons else WINDOW_DAYS
    signal_cols = tuple(args.signals) if args.signals else BENCHMARK_SIGNAL_COLS

    eval_horizons = parse_eval_horizons(args.eval_horizons)

    panels = build_freq_signal_panels(
        market=args.market,
        images_root=images_root,
        signals_root=signals_root,
        start=start,
        horizons=horizons,
    )

    written: list[Path] = []
    for signal_col in signal_cols:
        if signal_col not in BENCHMARK_SIGNAL_COLS:
            raise ValueError(f"unsupported signal={signal_col}")
        write_signal_parquets(panels, market=args.market, signal_col=signal_col)
        written.extend(
            backtest_signal(
                panels,
                market=args.market,
                signal_col=signal_col,
                ngroup=args.ngroup,
                eval_horizons=eval_horizons,
                fresh=args.fresh,
            )
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build trend benchmark signals and H1 backtest (CNN-aligned universe)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--market", choices=(MARKET_US, MARKET_CN), default=MARKET_US)
    common.add_argument("--ohlc", type=Path, default=None)
    common.add_argument("--signals-root", type=Path, default=None)
    common.add_argument("--images", type=Path, default=None)
    common.add_argument("--permno-limit", type=int, default=None)
    common.add_argument("--workers", type=int, default=None)
    common.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    common.add_argument("--fresh", action="store_true")

    p_sig = sub.add_parser("signals", parents=[common], help="per-stock daily signals")
    p_bt = sub.add_parser("backtest", parents=[common], help="join universe + H1 Excel")
    p_bt.add_argument("--start", type=str, default=None)
    p_bt.add_argument("--horizons", type=int, nargs="+", choices=WINDOW_DAYS, default=None)
    p_bt.add_argument("--signals", type=str, nargs="+", default=None)
    p_bt.add_argument("--ngroup", type=int, default=BACKTEST_N_GROUP)
    p_bt.add_argument(
        "--eval-horizons",
        type=int,
        nargs="+",
        default=[1],
        metavar="H",
        help="H1/H2/H3: signal at t, return at t+H rebalance periods "
        f"(default: 1; weekly grid: {' '.join(map(str, EVAL_HORIZONS))})",
    )

    p_tbl = sub.add_parser(
        "tables",
        parents=[common],
        help="Table V–VIII characteristics analysis (requires CNN 03/04)",
    )
    p_tbl.add_argument("--models", type=Path, default=None)

    p_all = sub.add_parser("all", parents=[common], help="signals then backtest")
    p_all.add_argument("--start", type=str, default=None)
    p_all.add_argument("--horizons", type=int, nargs="+", choices=WINDOW_DAYS, default=None)
    p_all.add_argument("--signals", type=str, nargs="+", default=None)
    p_all.add_argument("--ngroup", type=int, default=BACKTEST_N_GROUP)
    p_all.add_argument(
        "--eval-horizons",
        type=int,
        nargs="+",
        default=[1],
        metavar="H",
    )

    args = parser.parse_args()

    if args.command == "signals":
        run_signals(args)
        return
    if args.command == "backtest":
        run_backtest(args)
        return
    if args.command == "tables":
        paths = run_tables(args)
        for path in paths:
            log(f"wrote {path}")
        return
    if args.command == "all":
        run_signals(args)
        run_backtest(args)
        return
    raise ValueError(f"unknown command={args.command}")


if __name__ == "__main__":
    main()
