"""CLI pipeline for the STW 7,846 technical trading rule replication.

No data path is hard-coded here.  On the server, pass:
  --ohlc-root   path to PERMNO-partitioned daily OHLC parquet
  --universe    path to rebalance-date panel with PERMNO, Date, returns, caps
  --output-root path for the generated rule chunks and Sharpe summaries
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from backtest.stw_fast import STWSharpeConfig, fast_rule_chunk_sharpes
from config import BACKTEST_PERIODS_PER_YEAR, HORIZON_BACKTEST_DIR, MARKET_CN, MARKET_US
from data.parquet_io import permno_list, read_stock
from data.stw_rules import STWRule, manifest_frame, stw_rule_manifest, write_manifest
from data.stw_signals import compute_stock_rule_signals

DEFAULT_RULE_CHUNK_SIZE = 64
DEFAULT_STOCK_BATCH_SIZE = 16
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "08_stw_7846_rules"


def log(msg: str) -> None:
    print(msg, flush=True)


def horizon_dir(horizon: int) -> str:
    if horizon not in HORIZON_BACKTEST_DIR:
        raise ValueError(f"unsupported horizon={horizon}; expected one of {sorted(HORIZON_BACKTEST_DIR)}")
    return HORIZON_BACKTEST_DIR[horizon]


def pipeline_root(output_root: Path, market: str, horizon: int) -> Path:
    return Path(output_root) / market / horizon_dir(horizon)


def chunk_rules(rules: list[STWRule], chunk_size: int) -> list[list[STWRule]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [rules[i : i + chunk_size] for i in range(0, len(rules), chunk_size)]


def chunk_dir(root: Path, chunk_id: int) -> Path:
    return root / "rule_chunks" / f"chunk_{chunk_id:04d}"


def summary_chunk_path(root: Path, chunk_id: int) -> Path:
    return root / "sharpe_chunks" / f"summary_chunk_{chunk_id:04d}.parquet"


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


def load_universe(path: Path, *, start: str | None, permno_limit: int | None) -> pd.DataFrame:
    if path is None:
        raise ValueError(
            "--universe is required. TODO on server: pass a rebalance panel path with "
            "PERMNO, Date, forward return, and cap columns."
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"universe panel not found: {path}")
    universe = pd.read_parquet(path)
    missing = [c for c in ("PERMNO", "Date") if c not in universe.columns]
    if missing:
        raise KeyError(f"universe missing required columns {missing}")
    universe = universe.copy()
    universe["Date"] = pd.to_datetime(universe["Date"])
    universe["PERMNO"] = universe["PERMNO"].astype("int64")
    if start is not None:
        universe = universe[universe["Date"] >= pd.Timestamp(start)].copy()
    if permno_limit is not None:
        keep = sorted(universe["PERMNO"].unique())[:permno_limit]
        universe = universe[universe["PERMNO"].isin(keep)].copy()
    if universe.empty:
        raise ValueError("empty universe after filters")
    return universe.sort_values(["PERMNO", "Date"], kind="mergesort").reset_index(drop=True)


def write_manifest_command(args: argparse.Namespace) -> list[Path]:
    rules = stw_rule_manifest()
    root = pipeline_root(args.output_root, args.market, args.horizon)
    path = root / "stw_7846_manifest.csv"
    written = write_manifest(path, rules)
    counts = manifest_frame(rules)["family"].value_counts().sort_index().to_dict()
    log(f"wrote manifest {written} counts={counts}")
    return [written]


def _base_cols(universe: pd.DataFrame, ret_col: str) -> list[str]:
    cols = ["PERMNO", "Date", ret_col]
    for col in ("FloatCap", "TotalCap", "MarketCap"):
        if col in universe.columns and col not in cols:
            cols.append(col)
    missing = [c for c in ("PERMNO", "Date", ret_col) if c not in universe.columns]
    if missing:
        raise KeyError(
            f"universe missing {missing}. TODO on server: set --ret-col to the forward return "
            "column used for this horizon, e.g. Ret_5d, Ret_20d, or Ret_60d."
        )
    return cols


def build_signal_chunks(args: argparse.Namespace) -> list[Path]:
    if args.ohlc_root is None:
        raise ValueError("--ohlc-root is required. TODO on server: pass processed/<market>/ohlc_daily.")
    if not Path(args.ohlc_root).exists():
        raise FileNotFoundError(f"OHLC root not found: {args.ohlc_root}")

    ret_col = args.ret_col or f"Ret_{args.horizon}d"
    root = pipeline_root(args.output_root, args.market, args.horizon)
    if args.fresh and (root / "rule_chunks").exists():
        shutil.rmtree(root / "rule_chunks")
    root.mkdir(parents=True, exist_ok=True)

    rules = stw_rule_manifest()
    write_manifest(root / "stw_7846_manifest.csv", rules)
    rule_chunks = chunk_rules(rules, args.rule_chunk_size)
    universe = load_universe(args.universe, start=args.start, permno_limit=args.permno_limit)
    base_cols = _base_cols(universe, ret_col)

    available = set(int(p) for p in permno_list(args.ohlc_root))
    grouped = [(int(p), g[base_cols].copy()) for p, g in universe.groupby("PERMNO", sort=True) if int(p) in available]
    if not grouped:
        raise ValueError("no universe PERMNOs are present in --ohlc-root")

    written: list[Path] = []
    n_batches = (len(grouped) + args.stock_batch_size - 1) // args.stock_batch_size
    log(
        f"building STW signals stocks={len(grouped)} batches={n_batches} "
        f"rule_chunks={len(rule_chunks)} chunk_size={args.rule_chunk_size}"
    )

    for batch_id, start in enumerate(range(0, len(grouped), args.stock_batch_size)):
        batch = grouped[start : start + args.stock_batch_size]
        expected_parts = [chunk_dir(root, i) / f"part_{batch_id:05d}.parquet" for i in range(len(rule_chunks))]
        if not args.fresh and all(p.is_file() for p in expected_parts):
            log(f"skip batch {batch_id + 1}/{n_batches} (all chunk parts exist)")
            written.extend(expected_parts)
            continue

        buffers: list[list[pd.DataFrame]] = [[] for _ in rule_chunks]
        for permno, base in batch:
            stock_df = read_stock(args.ohlc_root, permno)
            as_ofs = pd.DatetimeIndex(base["Date"].unique())
            sig = compute_stock_rule_signals(stock_df, rules, as_ofs)
            sig.insert(0, "PERMNO", permno)
            merged = base.merge(sig, on=["PERMNO", "Date"], how="inner")

            for chunk_id, rule_chunk in enumerate(rule_chunks):
                rule_cols = [r.name for r in rule_chunk]
                buffers[chunk_id].append(merged[base_cols + rule_cols])

        for chunk_id, frames in enumerate(buffers):
            out_dir = chunk_dir(root, chunk_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"part_{batch_id:05d}.parquet"
            if out_path.exists() and args.fresh:
                out_path.unlink()
            if frames:
                pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
                written.append(out_path)
        log(f"wrote signal batch {batch_id + 1}/{n_batches}")

    return written


def run_backtest(args: argparse.Namespace) -> list[Path]:
    ret_col = args.ret_col or f"Ret_{args.horizon}d"
    root = pipeline_root(args.output_root, args.market, args.horizon)
    rules = stw_rule_manifest()
    rule_chunks = chunk_rules(rules, args.rule_chunk_size)

    config = STWSharpeConfig(
        date_col=args.date_col,
        ret_col=ret_col,
        periods_per_year=BACKTEST_PERIODS_PER_YEAR[args.horizon],
        ngroup=args.ngroup,
    )
    summary_dir = root / "sharpe_chunks"
    if args.fresh and summary_dir.exists():
        shutil.rmtree(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for chunk_id, rule_chunk in enumerate(rule_chunks):
        out_path = summary_chunk_path(root, chunk_id)
        if out_path.is_file() and not args.fresh:
            log(f"skip backtest chunk {chunk_id + 1}/{len(rule_chunks)}: {out_path}")
            written.append(out_path)
            continue

        in_dir = chunk_dir(root, chunk_id)
        if not in_dir.exists():
            raise FileNotFoundError(
                f"missing signal chunk {in_dir}; run `signals` first or copy server-generated chunks"
            )
        panel = pd.read_parquet(in_dir)
        signal_cols = [r.name for r in rule_chunk]
        weight_schemes: dict[str, str | None] = {"equal": None}
        for scheme, col in (("float", args.float_cap_col), ("total", args.total_cap_col)):
            if col is not None and col in panel.columns:
                weight_schemes[scheme] = col

        summary = fast_rule_chunk_sharpes(
            panel,
            signal_cols=signal_cols,
            config=config,
            weight_schemes=weight_schemes,
        )
        summary.insert(0, "horizon", args.horizon)
        summary.insert(1, "rebalance", horizon_dir(args.horizon))
        summary.insert(2, "chunk_id", chunk_id)
        summary.to_parquet(out_path, index=False)
        written.append(out_path)
        log(f"wrote backtest chunk {chunk_id + 1}/{len(rule_chunks)} -> {out_path}")

    combined = pd.concat([pd.read_parquet(p) for p in written], ignore_index=True)
    combined_path = root / "stw_7846_sharpes.parquet"
    combined_csv = root / "stw_7846_sharpes.csv"
    combined.to_parquet(combined_path, index=False)
    combined.to_csv(combined_csv, index=False)
    log(f"wrote combined summaries {combined_path} and {combined_csv}")

    figures_dir = root / "figures"
    cnn_sharpes = cnn_sharpes_from_args(args)
    fig_path = figures_dir / "figure8_stw_sharpe_distribution.png"
    from viz.stw_distribution import plot_stw_sharpe_distribution

    plot_stw_sharpe_distribution(
        combined,
        output_path=fig_path,
        title=f"STW 7,846 Technical Rules vs CNN ({args.market}, {horizon_dir(args.horizon)})",
        cnn_sharpes=cnn_sharpes,
        bins=args.plot_bins,
    )
    log(f"wrote distribution figure {fig_path}")

    figure_paths = [fig_path]
    for scheme, suffix in (
        ("equal", "equal_weight"),
        ("float", "float_value_weight"),
        ("total", "total_value_weight"),
    ):
        if scheme not in set(combined["weight_scheme"]):
            continue
        one_path = figures_dir / f"figure8_stw_sharpe_distribution_{suffix}.png"
        plot_stw_sharpe_distribution(
            combined,
            output_path=one_path,
            title=(
                f"STW 7,846 Technical Rules vs CNN "
                f"({args.market}, {horizon_dir(args.horizon)}, {scheme})"
            ),
            cnn_sharpes=cnn_sharpes,
            bins=args.plot_bins,
            schemes=[scheme],
        )
        figure_paths.append(one_path)
        log(f"wrote distribution figure {one_path}")

    return [*written, combined_path, combined_csv, *figure_paths]


def run_all(args: argparse.Namespace) -> list[Path]:
    written = []
    written.extend(write_manifest_command(args))
    written.extend(build_signal_chunks(args))
    written.extend(run_backtest(args))
    return written


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--market", choices=(MARKET_US, MARKET_CN), default=MARKET_US)
    parser.add_argument("--horizon", type=int, choices=tuple(HORIZON_BACKTEST_DIR), default=5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rule-chunk-size", type=int, default=DEFAULT_RULE_CHUNK_SIZE)
    parser.add_argument("--fresh", action="store_true")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ohlc-root", type=Path, default=None)
    parser.add_argument("--universe", type=Path, default=None)
    parser.add_argument("--ret-col", type=str, default=None)
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--float-cap-col", type=str, default="FloatCap")
    parser.add_argument("--total-cap-col", type=str, default="TotalCap")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--permno-limit", type=int, default=None)


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cnn-sharpe-equal", type=float, default=None)
    parser.add_argument("--cnn-sharpe-float", type=float, default=None)
    parser.add_argument("--cnn-sharpe-total", type=float, default=None)
    parser.add_argument("--plot-bins", type=int, default=80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate STW 7,846 technical-rule benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="write the 7,846-rule manifest")
    add_common(p_manifest)

    p_signals = sub.add_parser("signals", help="compute rebalance-date STW rule chunks")
    add_common(p_signals)
    add_data_args(p_signals)
    p_signals.add_argument("--stock-batch-size", type=int, default=DEFAULT_STOCK_BATCH_SIZE)

    p_backtest = sub.add_parser("backtest", help="compute fast H-L Sharpe distribution")
    add_common(p_backtest)
    add_data_args(p_backtest)
    add_plot_args(p_backtest)
    p_backtest.add_argument("--ngroup", type=int, default=10)

    p_all = sub.add_parser("all", help="manifest + signals + backtest")
    add_common(p_all)
    add_data_args(p_all)
    add_plot_args(p_all)
    p_all.add_argument("--stock-batch-size", type=int, default=DEFAULT_STOCK_BATCH_SIZE)
    p_all.add_argument("--ngroup", type=int, default=10)

    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest_command(args)
    elif args.command == "signals":
        build_signal_chunks(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "all":
        run_all(args)
    else:
        raise ValueError(f"unknown command={args.command}")


if __name__ == "__main__":
    main()
