"""CLI: OOS H1 quantile backtest with equal / float / total sheets."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
    CN_FACTOR_BACKTEST_DIR,
    HORIZON_BACKTEST_DIR,
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


def all_diagonal_configs() -> list[tuple[int, int]]:
    return [(d, h) for d in WINDOW_DAYS for h in WINDOW_DAYS]


def resolve_cnn_configs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.paper_cross:
        if args.all_configs or args.image_days is not None or args.horizon is not None:
            raise ValueError(
                "--paper-cross cannot be combined with --all-configs or --image-days/--horizon"
            )
        return list(PAPER_CROSS_CONFIGS)
    if args.image_days is not None and args.horizon is not None:
        return [(args.image_days, args.horizon)]
    if args.all_configs or args.image_days is None and args.horizon is None:
        return all_diagonal_configs()
    raise ValueError("provide both --image-days and --horizon for a single config")


def backtest_freq_dir(horizon: int) -> str:
    if horizon not in HORIZON_BACKTEST_DIR:
        raise ValueError(f"unsupported horizon={horizon}")
    return HORIZON_BACKTEST_DIR[horizon]


def summary_stem(tags: list[str], init_from: int | None, direct_signal: bool) -> str:
    if len(tags) == 1:
        raise ValueError("summary_stem requires multiple config tags")
    if init_from is not None:
        base = f"all_fromI{init_from}"
    else:
        base = "all"
    return f"direct_{base}" if direct_signal else base


def config_xlsx_path(
    market: str,
    horizon: int,
    tag: str,
    *,
    init_from: int | None,
    direct_signal: bool,
) -> Path:
    freq_dir = backtest_freq_dir(horizon)
    if init_from is not None:
        stem = f"{tag}_fromI{init_from}"
    else:
        stem = tag
    if direct_signal:
        stem = f"direct_{stem}"
    return BACKTEST_ROOT / market / freq_dir / f"{stem}_h1.xlsx"


def summary_xlsx_path(
    market: str,
    horizon: int,
    tags: list[str],
    *,
    init_from: int | None,
    direct_signal: bool,
) -> Path:
    freq_dir = backtest_freq_dir(horizon)
    stem = summary_stem(tags, init_from, direct_signal)
    return BACKTEST_ROOT / market / freq_dir / f"{stem}_h1.xlsx"


def default_cn_factor_xlsx_path(market: str, stem: str) -> Path:
    return BACKTEST_ROOT / market / CN_FACTOR_BACKTEST_DIR / f"{stem}_h1.xlsx"


def default_test_start(market: str) -> str:
    return market_sample_config(market).test_start


def group_configs_by_horizon(configs: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for image_days, horizon in configs:
        grouped[horizon].append((image_days, horizon))
    return dict(grouped)


def load_cnn_panel(
    *,
    market: str,
    image_days: int,
    horizon: int,
    models_root: Path,
    label_cache: dict[tuple[int, str], pd.DataFrame],
    start: str,
    init_from: int | None,
) -> tuple[pd.DataFrame, str]:
    tag = model_run_tag(image_days, horizon, init_from_image_days=init_from)
    pred = load_us_predictions(
        models_root,
        image_days,
        horizon,
        init_from_image_days=init_from,
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
    return panel, tag


def run_cnn_backtest(args: argparse.Namespace) -> list[Path]:
    configs = resolve_cnn_configs(args)
    market = args.market
    grouped = group_configs_by_horizon(configs)
    single_config = len(configs) == 1
    if args.output is not None and not single_config:
        raise ValueError("--output only allowed for a single --image-days/--horizon config")

    market_root = market_processed_dir(market)
    images_root = args.images if args.images is not None else market_root / "images"
    models_root = args.models if args.models is not None else market_root / "models"
    start = args.start if args.start is not None else default_test_start(market)
    direct_signal = args.direct_signal

    bundle_horizons: dict[tuple[int, str], set[int]] = {}
    for image_days, horizon in configs:
        freq = sample_freq_for_horizon(horizon)
        bundle_horizons.setdefault((image_days, freq), set()).add(horizon)
    label_cache = {
        (image_days, freq): load_image_labels(
            images_root, image_days, freq, tuple(sorted(horizons))
        )
        for (image_days, freq), horizons in bundle_horizons.items()
    }

    if args.fresh:
        for image_days, horizon in configs:
            tag = model_run_tag(
                image_days, horizon, init_from_image_days=args.init_from_image_days
            )
            ind = config_xlsx_path(
                market,
                horizon,
                tag,
                init_from=args.init_from_image_days,
                direct_signal=direct_signal,
            )
            if ind.is_file():
                ind.unlink()
        if not single_config:
            for horizon, h_configs in grouped.items():
                tags = [
                    model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
                    for d, h in h_configs
                ]
                summ = summary_xlsx_path(
                    market,
                    horizon,
                    tags,
                    init_from=args.init_from_image_days,
                    direct_signal=direct_signal,
                )
                if summ.is_file():
                    summ.unlink()

    if not single_config and not args.fresh:
        expected: list[Path] = []
        for image_days, horizon in configs:
            tag = model_run_tag(
                image_days, horizon, init_from_image_days=args.init_from_image_days
            )
            expected.append(
                config_xlsx_path(
                    market,
                    horizon,
                    tag,
                    init_from=args.init_from_image_days,
                    direct_signal=direct_signal,
                )
            )
        for horizon, h_configs in grouped.items():
            tags = [
                model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
                for d, h in h_configs
            ]
            expected.append(
                summary_xlsx_path(
                    market,
                    horizon,
                    tags,
                    init_from=args.init_from_image_days,
                    direct_signal=direct_signal,
                )
            )
        if all(p.is_file() for p in expected):
            for p in expected:
                log(f"skip backtest (exists): {p}")
            return expected

    scheme_rows_by_horizon: dict[int, dict[str, list[pd.DataFrame]]] = defaultdict(
        lambda: {"equal": [], "float": [], "total": []}
    )
    summary_dirty: dict[int, bool] = defaultdict(bool)
    written: list[Path] = []

    for image_days, horizon in configs:
        tag = model_run_tag(
            image_days, horizon, init_from_image_days=args.init_from_image_days
        )
        if market == MARKET_CN:
            spec = cn_cnn_spec(horizon)
        else:
            spec = us_spec(image_days, horizon)

        if args.output is not None:
            out_path = args.output
        else:
            out_path = config_xlsx_path(
                market,
                horizon,
                tag,
                init_from=args.init_from_image_days,
                direct_signal=direct_signal,
            )

        if single_config and out_path.is_file() and not args.fresh:
            log(f"skip config (exists): {out_path}")
            written.append(out_path)
            continue

        log(f"load {market.upper()} panel {tag}")
        panel, _ = load_cnn_panel(
            market=market,
            image_days=image_days,
            horizon=horizon,
            models_root=models_root,
            label_cache=label_cache,
            start=start,
            init_from=args.init_from_image_days,
        )
        log(f"{tag} n={len(panel)} dates={panel['Date'].nunique()}")
        tables = h1_perf_tables(
            panel,
            spec=spec,
            signal_cols=["p_up"],
            ngroup=args.ngroup,
            row_names=[tag],
            direct_signal=direct_signal,
        )

        write_individual = not out_path.is_file() or args.fresh
        if write_individual:
            path = write_h1_excel(tables, out_path)
            log(f"wrote {path}")
            written.append(path)
            if not single_config:
                summary_dirty[horizon] = True
        else:
            log(f"skip write config (exists): {out_path}")
            written.append(out_path)

        if not single_config:
            for scheme, frame in tables.items():
                scheme_rows_by_horizon[horizon][scheme].append(frame)

    if not single_config:
        for horizon in sorted(grouped):
            h_configs = grouped[horizon]
            tags = [
                model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
                for d, h in h_configs
            ]
            out_path = summary_xlsx_path(
                market,
                horizon,
                tags,
                init_from=args.init_from_image_days,
                direct_signal=direct_signal,
            )
            if out_path.is_file() and not args.fresh and not summary_dirty[horizon]:
                log(f"skip summary (exists): {out_path}")
                written.append(out_path)
                continue

            scheme_rows = scheme_rows_by_horizon[horizon]
            if not scheme_rows["equal"]:
                raise RuntimeError(f"missing tables for horizon={horizon} summary")
            combined = {
                scheme: pd.concat(frames, axis=0) for scheme, frames in scheme_rows.items()
            }
            path = write_h1_excel(combined, out_path)
            log(f"wrote {path}")
            written.append(path)

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
    out_path = args.output if args.output is not None else default_cn_factor_xlsx_path(
        "cn", stem
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
        direct_signal=args.direct_signal,
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
        help="all 9 I×R configs (default when --image-days/--horizon omitted)",
    )
    parser.add_argument(
        "--paper-cross",
        action="store_true",
        help="backtest 6 paper cross-frequency configs only",
    )
    parser.add_argument(
        "--direct-signal",
        action="store_true",
        help="sort deciles on raw p_up (paper Table I style); writes direct_*_h1.xlsx",
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
