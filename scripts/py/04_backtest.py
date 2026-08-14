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
    load_us_image_labels,
    load_us_predictions,
    merge_us_panel,
    read_table,
    write_h1_excel,
)
from backtest.markets import CN_SPEC, us_spec
from config import (
    BACKTEST_N_GROUP,
    BACKTEST_ROOT,
    IMAGES_ROOT,
    MODELS_ROOT,
    TEST_START,
    WINDOW_DAYS,
)
from models.dataset import model_run_tag


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_sig_cols(raw: str) -> list[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("empty --sig-cols")
    return cols


def resolve_us_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.all_configs:
        if args.image_days is not None or args.horizon is not None:
            raise ValueError("--all-configs cannot be combined with --image-days/--horizon")
        return [(d, h) for d in WINDOW_DAYS for h in WINDOW_DAYS]
    if args.image_days is None or args.horizon is None:
        raise ValueError("US backtest requires --image-days and --horizon, or --all-configs")
    return [(args.image_days, args.horizon)]


def default_xlsx_path(market: str, tags: list[str], init_from: int | None) -> Path:
    root = BACKTEST_ROOT / market
    if len(tags) == 1:
        return root / f"{tags[0]}_h1perf.xlsx"
    if init_from is not None:
        return root / f"all_fromI{init_from}_h1perf.xlsx"
    return root / "all_h1perf.xlsx"


def run_us(args: argparse.Namespace) -> Path:
    configs = resolve_us_configs(args)
    tags = [
        model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
        for d, h in configs
    ]
    out_path = args.output if args.output is not None else default_xlsx_path(
        "us", tags, args.init_from_image_days
    )
    if out_path.is_file() and not args.fresh:
        log(f"skip backtest (exists): {out_path}")
        return out_path
    if args.fresh and out_path.is_file():
        out_path.unlink()

    images_root = args.images if args.images is not None else IMAGES_ROOT
    models_root = args.models if args.models is not None else MODELS_ROOT
    horizons_by_image: dict[int, list[int]] = {}
    for image_days, horizon in configs:
        horizons_by_image.setdefault(image_days, []).append(horizon)
    label_cache = {
        image_days: load_us_image_labels(images_root, image_days, tuple(horizons))
        for image_days, horizons in horizons_by_image.items()
    }
    scheme_rows: dict[str, list[pd.DataFrame]] = {"equal": [], "float": [], "total": []}
    for image_days, horizon in configs:
        tag = model_run_tag(
            image_days, horizon, init_from_image_days=args.init_from_image_days
        )
        log(f"load US panel {tag}")
        pred = load_us_predictions(
            models_root,
            image_days,
            horizon,
            init_from_image_days=args.init_from_image_days,
        )
        panel = merge_us_panel(
            pred, label_cache[image_days], horizon=horizon, start=args.start
        )
        log(f"{tag} n={len(panel)} dates={panel['Date'].nunique()}")
        spec = us_spec(image_days, horizon)
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


def run_cn(args: argparse.Namespace) -> Path:
    if args.signals is None or args.returns is None:
        raise ValueError("China backtest requires --signals and --returns")
    if args.sig_cols is None:
        raise ValueError("China backtest requires --sig-cols")
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
    log(f"load CN signals n={len(signals)} returns n={len(returns)}")
    panel = load_cn_panel(
        signals,
        returns,
        spec=spec,
        lag=args.lag,
        universe=universe,
        start=args.start,
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
        help="US: backtest all image-days × horizon pairs",
    )
    parser.add_argument(
        "--init-from-image-days",
        type=int,
        default=None,
        choices=WINDOW_DAYS,
        help="US: read transfer-run ensemble predictions (fromI source window)",
    )
    parser.add_argument("--models", type=Path, default=None)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--signals", type=Path, default=None, help="China signal table")
    parser.add_argument("--returns", type=Path, default=None, help="China period-return table")
    parser.add_argument("--universe", type=Path, default=None, help="China universe filter")
    parser.add_argument("--sig-cols", type=str, default=None, help="comma-separated signal columns")
    parser.add_argument("--sigfile", type=str, default=None, help="China output name stem")
    parser.add_argument("--id-col", type=str, default=CN_SPEC.id_col)
    parser.add_argument("--date-col", type=str, default=CN_SPEC.date_col)
    parser.add_argument("--ret-col", type=str, default=CN_SPEC.ret_col)
    parser.add_argument("--float-cap-col", type=str, default=CN_SPEC.float_cap_col)
    parser.add_argument("--total-cap-col", type=str, default=CN_SPEC.total_cap_col)
    parser.add_argument("--lag", type=int, default=1, help="China: periods between signal and return")
    parser.add_argument("--periods-per-year", type=int, default=CN_SPEC.periods_per_year)
    parser.add_argument("--ngroup", type=int, default=BACKTEST_N_GROUP)
    parser.add_argument("--start", type=str, default=TEST_START)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.market == "us":
        run_us(args)
        return
    run_cn(args)


if __name__ == "__main__":
    main()
