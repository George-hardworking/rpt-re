"""CLI pipeline for the STW 7,846 technical trading rule replication (Figure 8).

Intermediate Sharpe chunks live on DATA_ROOT under processed/{market}/stw_7846_rules/.
Only Figure-8 PNGs are written to repo outputs/08_stw_7846_rules/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool, Value
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RESERVE_GIB = 16.0
MEM_PER_WORKER_GIB = 5.0
PROGRESS_EVERY = 100

from backtest.stw_fast import STWSharpeConfig, fast_rule_chunk_sharpes
from backtest.stw_io import (
    STW_UNIVERSE_PANEL,
    batch_fragments_complete,
    clear_horizon_scan_state,
    figure8_complete,
    load_figure8_cnn_sharpes,
    load_scan_checkpoint,
    load_stw_universe_table,
    mark_scan_batch_done,
    stw_data_root,
    stw_fragment_path,
    stw_fragments_dir,
    stw_market_figures_dir,
    stw_images_root,
    stw_market_spec,
    stw_ohlc_root,
    stw_universe_from_image_labels,
    stw_weight_schemes,
)
from config import (
    BACKTEST_PERIODS_PER_YEAR,
    BACKTEST_STW_ROOT,
    HORIZON_BACKTEST_DIR,
    MARKET_CN,
    MARKET_US,
    STW_DEFAULT_HORIZONS,
    market_sample_config,
)
from data.parquet_io import permno_list, read_stock
from data.stw_rules import STWRule, manifest_frame, stw_rule_manifest, write_manifest
from data.stw_signals import compute_stock_rule_signals
from utils.workers import resolve_workers

DEFAULT_RULE_CHUNK_SIZE = 64
DEFAULT_STOCK_BATCH_SIZE = 16

_PROGRESS = None
_TOTAL_STOCKS = None
_SKIP_STOCKS = None
_WORKER_OHLC: Path | None = None
_WORKER_UNIVERSE: dict[int, pd.DataFrame] = {}
_WORKER_RULES: list[STWRule] = []
_WORKER_PENDING_COLS: dict[int, list[str]] = {}
_WORKER_BASE_COLS: list[str] = []
_WORKER_FRAGMENT_DIR: Path | None = None
_WORKER_DATA_ROOT: Path | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def horizon_dir(horizon: int) -> str:
    if horizon not in HORIZON_BACKTEST_DIR:
        raise ValueError(
            f"unsupported horizon={horizon}; expected one of {sorted(HORIZON_BACKTEST_DIR)}"
        )
    return HORIZON_BACKTEST_DIR[horizon]


def chunk_rules(rules: list[STWRule], chunk_size: int) -> list[list[STWRule]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [rules[i : i + chunk_size] for i in range(0, len(rules), chunk_size)]


def summary_chunk_path(data_root: Path, chunk_id: int) -> Path:
    return data_root / "sharpe_chunks" / f"summary_chunk_{chunk_id:04d}.parquet"


def resolve_horizons(args: argparse.Namespace) -> tuple[int, ...]:
    if args.horizons is not None:
        return tuple(args.horizons)
    if args.horizon is not None:
        return (args.horizon,)
    return STW_DEFAULT_HORIZONS


def cnn_sharpes_from_args(args: argparse.Namespace) -> dict[str, float]:
    out: dict[str, float] = {}
    for scheme, attr in (
        ("equal", "cnn_sharpe_equal"),
        ("float", "cnn_sharpe_float"),
        ("total", "cnn_sharpe_total"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            out[scheme] = float(value)
    return out


def resolve_cnn_sharpes(args: argparse.Namespace, horizon: int) -> dict[str, float]:
    sharpes = load_figure8_cnn_sharpes(
        args.market,
        horizon,
        cnn_backtest_root=args.cnn_backtest_root,
        eval_horizon=args.cnn_eval_horizon,
        log=log,
    )
    sharpes.update(cnn_sharpes_from_args(args))
    return sharpes


def resolve_ohlc_root(args: argparse.Namespace) -> Path:
    root = stw_ohlc_root(args.market, args.ohlc_root)
    if not root.is_dir():
        raise FileNotFoundError(f"OHLC root not found: {root}")
    return root


def resolve_universe(args: argparse.Namespace, horizon: int) -> pd.DataFrame:
    start = args.start
    if start is None:
        start = market_sample_config(args.market).test_start
    if args.universe is not None:
        return load_stw_universe_table(
            args.universe,
            start=start,
            permno_limit=args.permno_limit,
        )
    images_root = stw_images_root(args.market, args.images_root)
    return stw_universe_from_image_labels(
        args.market,
        horizon,
        images_root=images_root,
        start=start,
        permno_limit=args.permno_limit,
    )


def _base_cols(universe: pd.DataFrame, ret_col: str, cap_cols: tuple[str, ...]) -> list[str]:
    cols = ["PERMNO", "Date", ret_col]
    for col in cap_cols:
        if col in universe.columns and col not in cols:
            cols.append(col)
    missing = [c for c in ("PERMNO", "Date", ret_col) if c not in universe.columns]
    if missing:
        raise KeyError(f"universe missing {missing}")
    return cols


def _init_stw_worker(
    universe_path: str,
    ohlc_path: str,
    fragment_dir: str,
    data_root: str,
    base_cols: list[str],
    pending_cols: dict[int, list[str]],
    progress,
    total_stocks: int,
    skip_stocks: int,
) -> None:
    global _WORKER_OHLC, _WORKER_UNIVERSE, _WORKER_RULES, _WORKER_PENDING_COLS
    global _WORKER_BASE_COLS, _WORKER_FRAGMENT_DIR, _WORKER_DATA_ROOT
    global _PROGRESS, _TOTAL_STOCKS, _SKIP_STOCKS
    universe = pd.read_parquet(universe_path)
    _WORKER_UNIVERSE = {
        int(p): g.copy() for p, g in universe.groupby("PERMNO", sort=True)
    }
    _WORKER_OHLC = Path(ohlc_path)
    _WORKER_FRAGMENT_DIR = Path(fragment_dir)
    _WORKER_DATA_ROOT = Path(data_root)
    _WORKER_RULES = stw_rule_manifest()
    _WORKER_BASE_COLS = base_cols
    _WORKER_PENDING_COLS = pending_cols
    _PROGRESS = progress
    _TOTAL_STOCKS = total_stocks
    _SKIP_STOCKS = skip_stocks


def _bump_stock_progress(n: int, worker_id: int, batch_id: int, batch_size: int) -> None:
    with _PROGRESS.get_lock():
        _PROGRESS.value += n
        done = _PROGRESS.value
    if done % PROGRESS_EVERY == 0 or done == _TOTAL_STOCKS:
        pct = 100.0 * done / _TOTAL_STOCKS
        log(
            f"progress {pct:5.1f}%  stocks {done}/{_TOTAL_STOCKS}  "
            f"skip={_SKIP_STOCKS}  worker={worker_id}  batch={batch_id + 1}  "
            f"batch_size={batch_size}"
        )


def process_stw_batch(task: tuple[int, list[int], int]) -> int:
    """Compute rule signals for one stock batch; write per-chunk fragment parquets."""
    batch_id, permnos, worker_id = task
    pending_ids = sorted(_WORKER_PENDING_COLS)
    if batch_fragments_complete(_WORKER_DATA_ROOT, batch_id, pending_ids):
        _bump_stock_progress(len(permnos), worker_id, batch_id, len(permnos))
        return batch_id

    _WORKER_FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_frames: dict[int, list[pd.DataFrame]] = {cid: [] for cid in pending_ids}

    for permno in permnos:
        base = _WORKER_UNIVERSE[int(permno)]
        stock_df = read_stock(_WORKER_OHLC, int(permno))
        as_ofs = pd.DatetimeIndex(base["Date"].unique())
        sig = compute_stock_rule_signals(stock_df, _WORKER_RULES, as_ofs)
        sig.insert(0, "PERMNO", permno)
        merged = base.merge(sig, on=["PERMNO", "Date"], how="inner")
        for chunk_id in pending_ids:
            cols = _WORKER_BASE_COLS + _WORKER_PENDING_COLS[chunk_id]
            chunk_frames[chunk_id].append(merged[cols])

    for chunk_id, frames in chunk_frames.items():
        if not frames:
            continue
        out_path = stw_fragment_path(_WORKER_DATA_ROOT, batch_id, chunk_id)
        pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)

    mark_scan_batch_done(_WORKER_DATA_ROOT, batch_id)
    _bump_stock_progress(len(permnos), worker_id, batch_id, len(permnos))
    return batch_id


def _merge_fragments_and_write_sharpe(
    *,
    data_root: Path,
    chunk_id: int,
    rule_chunk: list[STWRule],
    config: STWSharpeConfig,
    spec,
    horizon: int,
) -> Path:
    frag_dir = stw_fragments_dir(data_root)
    paths = sorted(frag_dir.glob(f"batch_*_chunk_{chunk_id:04d}.parquet"))
    if not paths:
        raise FileNotFoundError(f"no fragments for chunk_id={chunk_id} under {frag_dir}")
    panel = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    rule_cols = [r.name for r in rule_chunk]
    weight_schemes = stw_weight_schemes(panel, spec)
    summary = fast_rule_chunk_sharpes(
        panel,
        signal_cols=rule_cols,
        config=config,
        weight_schemes=weight_schemes,
    )
    summary.insert(0, "horizon", horizon)
    summary.insert(1, "rebalance", horizon_dir(horizon))
    summary.insert(2, "chunk_id", chunk_id)
    out_path = summary_chunk_path(data_root, chunk_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path, index=False)
    for p in paths:
        p.unlink()
    return out_path


def write_manifest_command(args: argparse.Namespace) -> list[Path]:
    written: list[Path] = []
    for horizon in resolve_horizons(args):
        rules = stw_rule_manifest()
        data_root = stw_data_root(args.market, horizon, args.processed_root)
        data_root.mkdir(parents=True, exist_ok=True)
        path = data_root / "stw_7846_manifest.csv"
        written.append(write_manifest(path, rules))
        counts = manifest_frame(rules)["family"].value_counts().sort_index().to_dict()
        log(f"horizon={horizon} wrote manifest {path} counts={counts}")
    return written


def compute_sharpe_chunks(args: argparse.Namespace) -> list[Path]:
    ohlc_root = resolve_ohlc_root(args)
    written: list[Path] = []

    for horizon in resolve_horizons(args):
        spec = stw_market_spec(args.market, horizon)
        ret_col = args.ret_col or spec.ret_col
        data_root = stw_data_root(args.market, horizon, args.processed_root)
        summary_dir = data_root / "sharpe_chunks"
        if args.fresh:
            if summary_dir.exists():
                shutil.rmtree(summary_dir)
            clear_horizon_scan_state(data_root)
        data_root.mkdir(parents=True, exist_ok=True)

        rules = stw_rule_manifest()
        write_manifest(data_root / "stw_7846_manifest.csv", rules)
        rule_chunks = chunk_rules(rules, args.rule_chunk_size)
        universe = resolve_universe(args, horizon)
        cap_cols = (spec.float_cap_col, spec.total_cap_col)
        base_cols = _base_cols(universe, ret_col, cap_cols)

        universe_path = data_root / STW_UNIVERSE_PANEL
        if args.fresh or not universe_path.is_file():
            universe.to_parquet(universe_path, index=False)

        available = set(int(p) for p in permno_list(ohlc_root))
        permnos = sorted(
            int(p)
            for p in universe["PERMNO"].unique()
            if int(p) in available
        )
        if not permnos:
            raise ValueError(
                f"no universe PERMNOs present in OHLC root={ohlc_root} horizon={horizon}"
            )

        batches: list[list[int]] = []
        for start in range(0, len(permnos), args.stock_batch_size):
            batches.append(permnos[start : start + args.stock_batch_size])
        n_batches = len(batches)

        config = STWSharpeConfig(
            date_col=args.date_col,
            ret_col=ret_col,
            periods_per_year=BACKTEST_PERIODS_PER_YEAR[horizon],
            ngroup=args.ngroup,
        )

        pending: list[int] = []
        for chunk_id in range(len(rule_chunks)):
            out_path = summary_chunk_path(data_root, chunk_id)
            if out_path.is_file() and not args.fresh:
                written.append(out_path)
            else:
                pending.append(chunk_id)

        if not pending:
            log(f"horizon={horizon} skip signals (all {len(rule_chunks)} sharpe chunks exist)")
            continue

        pending_cols = {
            chunk_id: [r.name for r in rule_chunks[chunk_id]] for chunk_id in pending
        }
        scan_done = load_scan_checkpoint(data_root)

        batch_tasks: list[tuple[int, list[int], int]] = []
        skip_stocks = 0
        for batch_id, batch_permnos in enumerate(batches):
            if (
                batch_id in scan_done
                and batch_fragments_complete(data_root, batch_id, pending)
            ):
                skip_stocks += len(batch_permnos)
                continue
            batch_tasks.append((batch_id, batch_permnos, -1))

        n_workers, diag = resolve_workers(
            len(batch_tasks),
            reserve_gib=args.reserve_gib,
            mem_per_worker_gib=MEM_PER_WORKER_GIB,
            override=args.workers,
        )
        log(
            f"horizon={horizon} STW scan stocks={len(permnos)} batches={n_batches} "
            f"pending_chunks={len(pending)} workers={n_workers} diag={diag} "
            f"data_root={data_root}"
        )

        progress = Value("i", skip_stocks)
        total_stocks = len(permnos)
        if skip_stocks > 0:
            pct = 100.0 * skip_stocks / total_stocks
            log(
                f"progress {pct:5.1f}%  stocks {skip_stocks}/{total_stocks}  "
                f"skip={skip_stocks}  (checkpoint resume)"
            )

        fragment_dir = stw_fragments_dir(data_root)
        fragment_dir.mkdir(parents=True, exist_ok=True)
        init_args = (
            str(universe_path),
            str(ohlc_root),
            str(fragment_dir),
            str(data_root),
            base_cols,
            pending_cols,
        )

        if batch_tasks:
            tasks = [
                (batch_id, batch_permnos, worker_id % n_workers)
                for worker_id, (batch_id, batch_permnos, _) in enumerate(batch_tasks)
            ]
            if n_workers == 1:
                _init_stw_worker(*init_args, progress, total_stocks, skip_stocks)
                for task in tasks:
                    process_stw_batch(task)
            else:
                with Pool(
                    n_workers,
                    initializer=_init_stw_worker,
                    initargs=(*init_args, progress, total_stocks, skip_stocks),
                ) as pool:
                    pool.map(process_stw_batch, tasks, chunksize=1)

        log(
            f"horizon={horizon} merge fragments -> sharpe chunks "
            f"pending={len(pending)}"
        )
        for i, chunk_id in enumerate(pending):
            out_path = summary_chunk_path(data_root, chunk_id)
            if out_path.is_file() and not args.fresh:
                written.append(out_path)
                continue
            out_path = _merge_fragments_and_write_sharpe(
                data_root=data_root,
                chunk_id=chunk_id,
                rule_chunk=rule_chunks[chunk_id],
                config=config,
                spec=spec,
                horizon=horizon,
            )
            written.append(out_path)
            pct = 100.0 * (i + 1) / len(pending)
            log(
                f"progress {pct:5.1f}%  sharpe_chunks {i + 1}/{len(pending)}  "
                f"horizon={horizon}  chunk_id={chunk_id}"
            )

        spill = data_root / "_tmp_signal_spill"
        if spill.exists():
            shutil.rmtree(spill)

    return written


def run_backtest(args: argparse.Namespace) -> list[Path]:
    horizons = resolve_horizons(args)
    fig_path = stw_market_figures_dir(args.market, args.figure_root) / (
        "figure8_stw_sharpe_distribution.png"
    )
    if figure8_complete(args.market, args.figure_root) and not args.fresh:
        log(f"skip backtest (figure exists): {fig_path}")
        return [fig_path]

    written: list[Path] = []
    panels: dict[int, pd.DataFrame] = {}
    cnn_by_horizon: dict[int, dict[str, float]] = {}

    for horizon in horizons:
        data_root = stw_data_root(args.market, horizon, args.processed_root)
        rules = stw_rule_manifest()
        rule_chunks = chunk_rules(rules, args.rule_chunk_size)

        chunk_paths: list[Path] = []
        for chunk_id in range(len(rule_chunks)):
            path = summary_chunk_path(data_root, chunk_id)
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing sharpe chunk {path}; run `signals` first horizon={horizon}"
                )
            chunk_paths.append(path)

        combined = pd.concat([pd.read_parquet(p) for p in chunk_paths], ignore_index=True)
        combined_path = data_root / "stw_7846_sharpes.parquet"
        combined_csv = data_root / "stw_7846_sharpes.csv"
        combined.to_parquet(combined_path, index=False)
        combined.to_csv(combined_csv, index=False)
        log(f"horizon={horizon} wrote combined summaries {combined_path}")

        panels[horizon] = combined
        cnn_by_horizon[horizon] = resolve_cnn_sharpes(args, horizon)
        written.extend([combined_path, combined_csv])

    from viz.stw_distribution import plot_stw_figure8_panel

    plot_stw_figure8_panel(
        panels,
        cnn_by_horizon,
        output_path=fig_path,
        market=args.market,
        bins=args.plot_bins,
    )
    log(f"wrote Figure-8 panel {fig_path}")
    written.append(fig_path)
    return written


def run_all(args: argparse.Namespace) -> list[Path]:
    written: list[Path] = []
    written.extend(write_manifest_command(args))
    written.extend(compute_sharpe_chunks(args))
    written.extend(run_backtest(args))
    return written


def add_worker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--market", choices=(MARKET_US, MARKET_CN), default=MARKET_US)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        choices=tuple(HORIZON_BACKTEST_DIR),
        default=None,
        help=f"rebalance horizons (default: {list(STW_DEFAULT_HORIZONS)})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=tuple(HORIZON_BACKTEST_DIR),
        default=None,
        help="single horizon override (use --horizons for Figure-8 default triple)",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="override DATA_ROOT parent for STW intermediates "
        "(default processed/{market}/stw_7846_rules/{freq})",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=BACKTEST_STW_ROOT,
        help="repo output root for Figure-8 PNGs only",
    )
    parser.add_argument("--rule-chunk-size", type=int, default=DEFAULT_RULE_CHUNK_SIZE)
    parser.add_argument("--fresh", action="store_true")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ohlc-root", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="optional rebalance parquet; default loads I20 image labels per horizon",
    )
    parser.add_argument("--ret-col", type=str, default=None)
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--permno-limit", type=int, default=None)


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cnn-backtest-root", type=Path, default=None)
    parser.add_argument("--cnn-eval-horizon", type=int, default=1)
    parser.add_argument("--cnn-sharpe-equal", type=float, default=None)
    parser.add_argument("--cnn-sharpe-float", type=float, default=None)
    parser.add_argument("--cnn-sharpe-total", type=float, default=None)
    parser.add_argument("--plot-bins", type=int, default=80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate STW 7,846 technical-rule benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="write the 7,846-rule manifest")
    add_common(p_manifest)

    p_signals = sub.add_parser(
        "signals",
        help="parallel stock scan -> per-chunk H-L Sharpe on DATA_ROOT",
    )
    add_common(p_signals)
    add_data_args(p_signals)
    add_worker_args(p_signals)
    p_signals.add_argument("--stock-batch-size", type=int, default=DEFAULT_STOCK_BATCH_SIZE)
    p_signals.add_argument("--ngroup", type=int, default=10)

    p_backtest = sub.add_parser("backtest", help="combine sharpe chunks and write Figure-8 PNGs")
    add_common(p_backtest)
    add_data_args(p_backtest)
    add_plot_args(p_backtest)

    p_all = sub.add_parser("all", help="manifest + signals + backtest")
    add_common(p_all)
    add_data_args(p_all)
    add_plot_args(p_all)
    add_worker_args(p_all)
    p_all.add_argument("--stock-batch-size", type=int, default=DEFAULT_STOCK_BATCH_SIZE)
    p_all.add_argument("--ngroup", type=int, default=10)

    args = parser.parse_args()
    if hasattr(args, "cnn_backtest_root") and args.cnn_backtest_root is None:
        from config import BACKTEST_CNN_ROOT

        args.cnn_backtest_root = BACKTEST_CNN_ROOT

    if args.command == "manifest":
        write_manifest_command(args)
    elif args.command == "signals":
        compute_sharpe_chunks(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "all":
        run_all(args)
    else:
        raise ValueError(f"unknown command={args.command}")


if __name__ == "__main__":
    main()
