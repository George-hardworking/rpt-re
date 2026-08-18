"""CLI: OOS quantile backtest with equal / float / total sheets (H1/H2/H3 eval horizons)."""

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
from backtest.eval_horizon import align_panel_eval_horizon, backtest_output_stem, parse_eval_horizons
from backtest.io import (
    filter_top_n_by_cap,
    load_cn_panel,
    load_image_labels,
    load_us_predictions,
    merge_cnn_panel,
    read_table,
    write_h1_excel,
)
from backtest.markets import CN_SPEC, cn_cnn_spec, us_spec
from config import (
    BACKTEST_CNN_ROOT,
    BACKTEST_CNN_TOP500_FLOAT_ROOT,
    BACKTEST_CNN_TOP500_ROOT,
    BACKTEST_CN_FACTOR_ROOT,
    BACKTEST_N_GROUP,
    CN_FACTOR_BACKTEST_DIR,
    EVAL_HORIZONS,
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


def cnn_output_root(top_n: int | None, top_n_cap: str = "total") -> Path:
    if top_n is None:
        return BACKTEST_CNN_ROOT
    if top_n_cap == "float":
        return BACKTEST_CNN_TOP500_FLOAT_ROOT
    return BACKTEST_CNN_TOP500_ROOT


def config_xlsx_path(
    market: str,
    horizon: int,
    tag: str,
    *,
    eval_horizon: int,
    init_from: int | None,
    direct_signal: bool,
    output_root: Path,
) -> Path:
    freq_dir = backtest_freq_dir(horizon)
    if init_from is not None:
        base = f"{tag}_fromI{init_from}"
    else:
        base = tag
    stem = backtest_output_stem(base, eval_horizon, direct_signal=direct_signal)
    return output_root / market / freq_dir / f"{stem}.xlsx"


def summary_xlsx_path(
    market: str,
    horizon: int,
    tags: list[str],
    *,
    eval_horizon: int,
    init_from: int | None,
    direct_signal: bool,
    output_root: Path,
) -> Path:
    freq_dir = backtest_freq_dir(horizon)
    base = summary_stem(tags, init_from, direct_signal)
    stem = backtest_output_stem(base, eval_horizon, direct_signal=False)
    return output_root / market / freq_dir / f"{stem}.xlsx"


def default_cn_factor_xlsx_path(market: str, stem: str, eval_horizon: int) -> Path:
    return (
        BACKTEST_CN_FACTOR_ROOT
        / market
        / CN_FACTOR_BACKTEST_DIR
        / f"{stem}_h{eval_horizon}.xlsx"
    )


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
    eval_horizons = parse_eval_horizons(args.eval_horizons)
    market = args.market
    grouped = group_configs_by_horizon(configs)
    single_config = len(configs) == 1
    if args.output is not None and (not single_config or len(eval_horizons) != 1):
        raise ValueError("--output only allowed for one config and one --eval-horizons value")

    market_root = market_processed_dir(market)
    images_root = args.images if args.images is not None else market_root / "images"
    models_root = args.models if args.models is not None else market_root / "models"
    start = args.start if args.start is not None else default_test_start(market)
    direct_signal = args.direct_signal
    top_n = args.top_n
    top_n_cap = args.top_n_cap
    if top_n is None and top_n_cap != "total":
        raise ValueError("--top-n-cap float requires --top-n")
    if top_n_cap == "float" and market != MARKET_CN:
        raise ValueError("--top-n-cap float only supported for --market cn")
    output_root = cnn_output_root(top_n, top_n_cap)

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
            for eval_h in eval_horizons:
                ind = config_xlsx_path(
                    market,
                    horizon,
                    tag,
                    eval_horizon=eval_h,
                    init_from=args.init_from_image_days,
                    direct_signal=direct_signal,
                    output_root=output_root,
                )
                if ind.is_file():
                    ind.unlink()
        if not single_config:
            for horizon, h_configs in grouped.items():
                tags = [
                    model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
                    for d, h in h_configs
                ]
                for eval_h in eval_horizons:
                    summ = summary_xlsx_path(
                        market,
                        horizon,
                        tags,
                        eval_horizon=eval_h,
                        init_from=args.init_from_image_days,
                        direct_signal=direct_signal,
                        output_root=output_root,
                    )
                    if summ.is_file():
                        summ.unlink()

    if not single_config and not args.fresh:
        expected: list[Path] = []
        for image_days, horizon in configs:
            tag = model_run_tag(
                image_days, horizon, init_from_image_days=args.init_from_image_days
            )
            for eval_h in eval_horizons:
                expected.append(
                    config_xlsx_path(
                        market,
                        horizon,
                        tag,
                        eval_horizon=eval_h,
                        init_from=args.init_from_image_days,
                        direct_signal=direct_signal,
                        output_root=output_root,
                    )
                )
        for horizon, h_configs in grouped.items():
            tags = [
                model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
                for d, h in h_configs
            ]
            for eval_h in eval_horizons:
                expected.append(
                    summary_xlsx_path(
                        market,
                        horizon,
                        tags,
                        eval_horizon=eval_h,
                        init_from=args.init_from_image_days,
                        direct_signal=direct_signal,
                        output_root=output_root,
                    )
                )
        if all(p.is_file() for p in expected):
            for p in expected:
                log(f"skip backtest (exists): {p}")
            return expected

    scheme_rows: dict[tuple[int, int], dict[str, list[pd.DataFrame]]] = defaultdict(
        lambda: {"equal": [], "float": [], "total": []}
    )
    summary_dirty: dict[tuple[int, int], bool] = defaultdict(bool)
    written: list[Path] = []

    for image_days, horizon in configs:
        tag = model_run_tag(
            image_days, horizon, init_from_image_days=args.init_from_image_days
        )
        if market == MARKET_CN:
            spec = cn_cnn_spec(horizon)
        else:
            spec = us_spec(image_days, horizon)

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
        if top_n is not None:
            panel = filter_top_n_by_cap(
                panel, spec=spec, top_n=top_n, cap_kind=top_n_cap
            )
            log(
                f"{tag} top-{top_n} ({top_n_cap}) n={len(panel)} "
                f"dates={panel['Date'].nunique()}"
            )

        for eval_h in eval_horizons:
            if args.output is not None:
                out_path = args.output
            else:
                out_path = config_xlsx_path(
                    market,
                    horizon,
                    tag,
                    eval_horizon=eval_h,
                    init_from=args.init_from_image_days,
                    direct_signal=direct_signal,
                    output_root=output_root,
                )

            if out_path.is_file() and not args.fresh:
                log(f"skip config (exists): {out_path}")
                written.append(out_path)
                continue

            aligned = align_panel_eval_horizon(panel, spec, eval_h)
            tables = h1_perf_tables(
                aligned,
                spec=spec,
                signal_cols=["p_up"],
                ngroup=args.ngroup,
                row_names=[tag],
                direct_signal=direct_signal,
            )
            path = write_h1_excel(tables, out_path)
            log(f"wrote {path}")
            written.append(path)
            if not single_config:
                summary_dirty[(horizon, eval_h)] = True
                for scheme, frame in tables.items():
                    scheme_rows[(horizon, eval_h)][scheme].append(frame)

    if not single_config:
        for horizon in sorted(grouped):
            h_configs = grouped[horizon]
            tags = [
                model_run_tag(d, h, init_from_image_days=args.init_from_image_days)
                for d, h in h_configs
            ]
            for eval_h in eval_horizons:
                out_path = summary_xlsx_path(
                    market,
                    horizon,
                    tags,
                    eval_horizon=eval_h,
                    init_from=args.init_from_image_days,
                    direct_signal=direct_signal,
                    output_root=output_root,
                )
                if out_path.is_file() and not args.fresh and not summary_dirty[(horizon, eval_h)]:
                    log(f"skip summary (exists): {out_path}")
                    written.append(out_path)
                    continue

                rows = scheme_rows[(horizon, eval_h)]
                if not rows["equal"]:
                    raise RuntimeError(
                        f"missing tables for horizon={horizon} eval_h={eval_h} summary"
                    )
                combined = {
                    scheme: pd.concat(frames, axis=0) for scheme, frames in rows.items()
                }
                path = write_h1_excel(combined, out_path)
                log(f"wrote {path}")
                written.append(path)

    return written


def run_cn_factor(args: argparse.Namespace) -> list[Path]:
    if args.signals is None or args.returns is None:
        raise ValueError("China factor backtest requires --signals and --returns")
    if args.sig_cols is None:
        raise ValueError("China factor backtest requires --sig-cols")
    sig_cols = parse_sig_cols(args.sig_cols)
    eval_horizons = parse_eval_horizons(args.eval_horizons)
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
    signals = read_table(args.signals)
    returns = read_table(args.returns)
    universe = read_table(args.universe) if args.universe is not None else None
    start = args.start if args.start is not None else default_test_start(MARKET_CN)
    log(f"load CN signals n={len(signals)} returns n={len(returns)}")

    written: list[Path] = []
    for eval_h in eval_horizons:
        if args.output is not None:
            if len(eval_horizons) != 1:
                raise ValueError("--output requires a single --eval-horizons value")
            out_path = args.output
        else:
            out_path = default_cn_factor_xlsx_path("cn", stem, eval_h)
        if out_path.is_file() and not args.fresh:
            log(f"skip backtest (exists): {out_path}")
            written.append(out_path)
            continue
        if args.fresh and out_path.is_file():
            out_path.unlink()

        panel = load_cn_panel(
            signals,
            returns,
            spec=spec,
            lag=eval_h,
            universe=universe,
            start=start,
        )
        log(f"CN panel eval_h={eval_h} n={len(panel)} dates={panel[spec.date_col].nunique()}")
        tables = h1_perf_tables(
            panel,
            spec=spec,
            signal_cols=sig_cols,
            ngroup=args.ngroup,
            direct_signal=args.direct_signal,
        )
        path = write_h1_excel(tables, out_path)
        log(f"wrote {path}")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantile backtest; Hk = return at t+k rebalance periods; Excel equal/float/total"
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
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="keep top N stocks by formation-date cap; writes outputs/04_backtest/cnn_top500/",
    )
    parser.add_argument(
        "--top-n-cap",
        choices=("total", "float"),
        default="total",
        help="CN only: rank top-N on TotalCap (default) or FloatCap → outputs/04_backtest/cnn_top500_float/",
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
    parser.add_argument(
        "--eval-horizons",
        type=int,
        nargs="+",
        default=[1],
        metavar="H",
        help="evaluation horizons H1/H2/H3: signal at t, return at t+H rebalance periods "
        f"(default: 1; typical weekly grid: {' '.join(map(str, EVAL_HORIZONS))})",
    )
    parser.add_argument("--lag", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--periods-per-year", type=int, default=CN_SPEC.periods_per_year)
    parser.add_argument("--ngroup", type=int, default=BACKTEST_N_GROUP)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.lag is not None:
        args.eval_horizons = [args.lag]

    if args.market == MARKET_CN and args.signals is not None:
        run_cn_factor(args)
        return
    run_cnn_backtest(args)


if __name__ == "__main__":
    main()
