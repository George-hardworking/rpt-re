"""CLI: OOS H1 quantile backtest with equal / float / total sheets."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import h1_perf_tables
from backtest.io import (
    load_cn_panel,
    load_image_labels,
    load_us_predictions,
    merge_cnn_panel,
    read_table,
    write_h1_excel,
)
from backtest.markets import CN_SPEC, cn_cnn_spec, us_spec
from config import (
    BACKTEST_N_GROUP,
    BACKTEST_ROOT,
    MARKET_CN,
    PAPER_CROSS_CONFIGS,
    TEST_START,
    WINDOW_DAYS,
    market_processed_dir,
    market_sample_config,
    sample_freq_for_horizon,
)
from models.dataset import model_run_tag


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_sig_cols(raw: str) -> list[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("empty --sig-cols")
    return cols


def resolve_cnn_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.paper_cross:
        if args.all_configs or args.image_days is not None or args.horizon is not None:
            raise ValueError(
                "--paper-cross cannot be combined with --all-configs or --image-days/--horizon"
            )
        return list(PAPER_CROSS_CONFIGS)
    if args.all_configs:
        if args.image_days is not None or args.horizon is not None:
            raise ValueError("--all-configs cannot be combined with --image-days/--horizon")
        return [(d, h) for d in WINDOW_DAYS for h in WINDOW_DAYS]
    if args.image_days is None or args.horizon is None:
        raise ValueError(
            "CNN backtest requires --image-days and --horizon, --all-configs, or --paper-cross"
        )
    return [(args.image_days, args.horizon)]


def default_xlsx_path(market: str, tags: list[str], init_from: int | None) -> Path:
    root = BACKTEST_ROOT / market
    if len(tags) == 1:
        return root / f"{tags[0]}_h1perf.xlsx"
    if init_from is not None:
        return root / f"all_fromI{init_from}_h1perf.xlsx"
    return root / "all_h1perf.xlsx"


def default_test_start(market: str) -> str:
    return market_sample_config(market).test_start


def run_cnn_backtest(args: argparse.Namespace) -> Path:
    configs = resolve_cnn_configs(args)
    market = args.market
    tags = [
        model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
        for d, h in configs
    ]
    out_path = args.output if args.output is not None else default_xlsx_path(
        market, tags, args.init_from_image_days
    )
    if out_path.is_file() and not args.fresh:
        log(f"skip backtest (exists): {out_path}")
        return out_path
    if args.fresh and out_path.is_file():
        out_path.unlink()

    market_root = market_processed_dir(market)
    images_root = args.images if args.images is not None else market_root / "images"
    models_root = args.models if args.models is not None else market_root / "models"
    start = args.start if args.start is not None else default_test_start(market)

    bundle_horizons: dict[tuple[int, str], set[int]] = {}
    for image_days, horizon in configs:
        freq = sample_freq_for_horizon(horizon)
        bundle_horizons.setdefault((image_days, freq), set()).add(horizon)
    label_cache = {
        key: load_image_labels(images_root, image_days, freq, tuple(sorted(horizons)))
        for (image_days, freq), horizons in bundle_horizons.items()
    }
    scheme_rows: dict[str, list[pd.DataFrame]] = {"equal": [], "float": [], "total": []}
    for image_days, horizon in configs:
        tag = model_run_tag(
            image_days, horizon, init_from_image_days=args.init_from_image_days
        )
        log(f"load {market.upper()} panel {tag}")
        pred = load_us_predictions(
            models_root,
            image_days,
            horizon,
            init_from_image_days=args.init_from_image_days,
        )
        freq = sample_freq_for_horizon(horizon)
        if market == MARKET_CN:
            spec = cn_cnn_spec(horizon)
        else:
            spec = us_spec(image_days, horizon)
        panel = merge_cnn_panel(
            pred,
            label_cache[(image_days, freq)],
            horizon=horizon,
            spec=spec,
            start=start,
        )
        log(f"{tag} n={len(panel)} dates={panel['Date'].nunique()}")
        tables = h1_perf_tables(
            panel,
            spec=spec,
            signal_cols=["p_up"],
            ngroup=args.ngroup,
            row_names=[tag],
        )
        for scheme, frame in tables.items():
            scheme_rows[scheme].append(frame)

    combined = {scheme: pd.concat(frames, axis=0) for scheme, frames in scheme_rows.items()}
    written = write_h1_excel(combined, out_path)
    log(f"wrote {written}")
    return written


def run_cn_factor(args: argparse.Namespace) -> Path:
    if args.signals is None or args.returns is None:
        raise ValueError("China factor backtest requires --signals and --returns")
    if args.sig_cols is None:
        raise ValueError("China factor backtest requires --sig-cols")
    sig_cols = parse_sig_cols(args.sig_cols)
    spec = replace(
        CN_SPEC,
        id_col=args.id_col,
        date_col=args.date_col,
        ret_col=args.ret_col,
        float_cap_col=args.float_cap_col,
        total_cap_col=args.total_cap_col,
        periods_per_year=args.periods_per_year,
    )
    stem = args.sigfile if args.sigfile else Path(args.signals).stem
    out_path = args.output if args.output is not None else default_xlsx_path(
        "cn", [stem], None
    )
    if out_path.is_file() and not args.fresh:
        log(f"skip backtest (exists): {out_path}")
        return out_path
    if args.fresh and out_path.is_file():
        out_path.unlink()

    signals = read_table(args.signals)
    returns = read_table(args.returns)
    universe = read_table(args.universe) if args.universe is not None else None
    start = args.start if args.start is not None else default_test_start(MARKET_CN)
    log(f"load CN signals n={len(signals)} returns n={len(returns)}")
    panel = load_cn_panel(
        signals,
        returns,
        spec=spec,
        lag=args.lag,
        universe=universe,
        start=start,
    )
    log(f"CN panel n={len(panel)} dates={panel[spec.date_col].nunique()}")
    tables = h1_perf_tables(
        panel,
        spec=spec,
        signal_cols=sig_cols,
        ngroup=args.ngroup,
    )
    written = write_h1_excel(tables, out_path)
    log(f"wrote {written}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H1 quantile backtest; writes Excel sheets equal/float/total"
    )
    parser.add_argument("--market", choices=("us", "cn"), default="us")
    parser.add_argument("--image-days", type=int, default=None, choices=WINDOW_DAYS)
    parser.add_argument("--horizon", type=int, default=None, choices=WINDOW_DAYS)
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="backtest all image-days × horizon pairs",
    )
    parser.add_argument(
        "--paper-cross",
        action="store_true",
        help="backtest 6 paper cross-frequency configs only",
    )
    parser.add_argument(
        "--init-from-image-days",
        type=int,
        default=None,
        choices=WINDOW_DAYS,
        help="read transfer-run ensemble predictions (fromI source window)",
    )
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--signals", type=Path, default=None, help="China factor signal table")
    parser.add_argument("--returns", type=Path, default=None, help="China period-return table")
    parser.add_argument("--universe", type=Path, default=None, help="China universe filter")
    parser.add_argument("--sig-cols", type=str, default=None, help="comma-separated signal columns")
    parser.add_argument("--sigfile", type=str, default=None, help="China output name stem")
    parser.add_argument("--id-col", type=str, default=CN_SPEC.id_col)
    parser.add_argument("--date-col", type=str, default=CN_SPEC.date_col)
    parser.add_argument("--ret-col", type=str, default=CN_SPEC.ret_col)
    parser.add_argument("--float-cap-col", type=str, default=CN_SPEC.float_cap_col)
    parser.add_argument("--total-cap-col", type=str, default=CN_SPEC.total_cap_col)
    parser.add_argument("--lag", type=int, default=1, help="China factor: periods between signal and return")
    parser.add_argument("--periods-per-year", type=int, default=CN_SPEC.periods_per_year)
    parser.add_argument("--ngroup", type=int, default=BACKTEST_N_GROUP)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.market == MARKET_CN and args.signals is not None:
        run_cn_factor(args)
        return
    run_cnn_backtest(args)


if __name__ == "__main__":
    main()
