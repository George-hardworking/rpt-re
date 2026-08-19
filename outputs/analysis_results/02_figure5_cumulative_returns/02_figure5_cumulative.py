"""02 — Figure 5 style cumulative H–L returns (raw + vol-adjusted) for US and CN.

Read-only: loads step-04 CNN predictions, step-05 benchmark panels, akshare SPY/510300.
Produces four PNGs in this directory (weekly R5, equal-weight H–L + CNN VW, eval H1).

Run (from repo root, 5020_env):
  python "outputs/analysis_results/02_figure5_cumulative_returns/02_figure5_cumulative.py"
  python "outputs/analysis_results/02_figure5_cumulative_returns/02_figure5_cumulative.py" --market us
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import hl_return_series
from backtest.eval_horizon import align_panel_eval_horizon
from backtest.io import load_image_labels, load_us_predictions, merge_cnn_panel
from backtest.markets import cn_cnn_spec, us_spec
from config import (
    MARKET_CN,
    MARKET_US,
    benchmark_signals_dir,
    market_processed_dir,
    market_sample_config,
    sample_freq_for_horizon,
)
from data.etf_returns import (
    daily_returns,
    fetch_510300_daily,
    fetch_spy_daily,
    forward_holding_returns,
    period_volatility,
    scale_to_benchmark_vol,
)

HERE = Path(__file__).resolve().parent
STEM = "figure5_cumulative"
HORIZON = 5
EVAL_HORIZON = 1
# Square plot box plus a thin strip for tick labels; canvas matches the figure.
PLOT_IN = 2.05
LEFT_IN = 0.38
RIGHT_IN = 0.03
BOTTOM_IN = 0.22
TOP_IN = 0.03
FIG_W = PLOT_IN + LEFT_IN + RIGHT_IN
FIG_H = PLOT_IN + BOTTOM_IN + TOP_IN
FIG_DPI = 1200
FONT_FAMILY = "Times New Roman"
NIMBUS_ROMAN_PATH = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"

CNN_CONFIGS: tuple[tuple[int, str], ...] = (
    (5, "I5/R5"),
    (20, "I20/R5"),
    (60, "I60/R5"),
)

BENCHMARK_CONFIGS: tuple[tuple[str, str], ...] = (
    ("MOM/R5", "MOM"),
    ("STR/R5", "REV1m_STR"),
    ("WSTR/R5", "REV1w_WSTR"),
    ("TREND/R5", "TREND_HZZ"),
)

# User-specified colors; VW lines reuse CNN colors with dashed style.
SERIES_COLOR: dict[str, str] = {
    "I5/R5": "#1f77b4",
    "I20/R5": "#d62728",
    "I60/R5": "#000000",
    "MOM/R5": "#2ca02c",
    "STR/R5": "#98e6b9",
    "WSTR/R5": "#9467bd",
    "TREND/R5": "#8b0000",
    "SPY": "#b8860b",
    "510300": "#b8860b",
}

BENCHMARK_ETF: dict[str, tuple[str, str]] = {
    MARKET_US: ("SPY", "SPY"),
    MARKET_CN: ("510300", "CSI300 ETF (510300)"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def period_years(market: str) -> tuple[int, int]:
    cfg = market_sample_config(market)
    y0 = pd.Timestamp(cfg.test_start).year
    y1 = pd.Timestamp(cfg.sample_end).year
    return y0, y1


def period_bounds(market: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = market_sample_config(market)
    return pd.Timestamp(cfg.test_start), pd.Timestamp(cfg.sample_end)


def cum_log_returns(returns: pd.Series) -> pd.Series:
    r = returns.astype(np.float64).sort_index()
    return np.log1p(r).cumsum()


def vw_label(ew_label: str) -> str:
    return f"{ew_label} (VW)"


def series_style(label: str) -> dict[str, object]:
    base = label.replace(" (VW)", "")
    color = SERIES_COLOR.get(base, SERIES_COLOR.get(label, "#333333"))
    ls = "--" if "(VW)" in label else "-"
    lw = 0.55 if "(VW)" in label else 0.75
    return {"color": color, "ls": ls, "lw": lw}


def load_cnn_panel_aligned(
    market: str,
    image_days: int,
    label_cache: dict,
) -> tuple[pd.DataFrame, object]:
    market_root = market_processed_dir(market)
    models_root = market_root / "models"
    pred = load_us_predictions(models_root, image_days, HORIZON)
    freq = sample_freq_for_horizon(HORIZON)
    if market == MARKET_CN:
        spec = cn_cnn_spec(HORIZON)
    else:
        spec = us_spec(image_days, HORIZON)
    start, end = period_bounds(market)
    panel = merge_cnn_panel(
        pred,
        label_cache[(image_days, freq)],
        horizon=HORIZON,
        spec=spec,
        start=start.strftime("%Y-%m-%d"),
    )
    panel = panel[(panel["Date"] >= start) & (panel["Date"] <= end)]
    panel = align_panel_eval_horizon(panel, spec, EVAL_HORIZON)
    return panel, spec


def load_cnn_hl(
    market: str,
    image_days: int,
    label_cache: dict,
    *,
    scheme: str,
) -> pd.Series:
    panel, spec = load_cnn_panel_aligned(market, image_days, label_cache)
    return hl_return_series(panel, spec=spec, signal_col="p_up", scheme=scheme)


def load_benchmark_hl(market: str, signal_col: str) -> pd.Series:
    path = benchmark_signals_dir(signal_col, market, HORIZON) / "signals.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run ./scripts/sh/05_benchmark_signals.sh backtest --market {market}"
        )
    panel = pd.read_parquet(path)
    panel["Date"] = pd.to_datetime(panel["Date"])
    start, end = period_bounds(market)
    panel = panel[(panel["Date"] >= start) & (panel["Date"] <= end)]
    if market == MARKET_CN:
        spec = cn_cnn_spec(HORIZON)
    else:
        spec = us_spec(HORIZON, HORIZON)
    panel = align_panel_eval_horizon(panel, spec, EVAL_HORIZON)
    return hl_return_series(panel, spec=spec, signal_col=signal_col, scheme="equal")


def load_all_hl(market: str) -> dict[str, pd.Series]:
    market_root = market_processed_dir(market)
    images_root = market_root / "images"
    freq = sample_freq_for_horizon(HORIZON)
    label_cache: dict[tuple[int, str], pd.DataFrame] = {}
    for image_days, label in CNN_CONFIGS:
        key = (image_days, freq)
        if key not in label_cache:
            label_cache[key] = load_image_labels(images_root, image_days, freq, (HORIZON,))

    series: dict[str, pd.Series] = {}
    for image_days, label in CNN_CONFIGS:
        series[label] = load_cnn_hl(market, image_days, label_cache, scheme="equal")
        vw = load_cnn_hl(market, image_days, label_cache, scheme="total")
        series[vw_label(label)] = vw
        log(f"{market} {label} EW={len(series[label])} VW={len(vw)}")
    for label, signal_col in BENCHMARK_CONFIGS:
        series[label] = load_benchmark_hl(market, signal_col)
        log(f"{market} {label}: n={len(series[label])}")
    return series


def load_benchmark_etf_holding_returns(
    market: str,
    formation_dates: pd.DatetimeIndex,
) -> pd.Series:
    start, end = period_bounds(market)
    pad_start = (start - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    pad_end = (end + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    if market == MARKET_US:
        etf = fetch_spy_daily(pad_start, pad_end)
    elif market == MARKET_CN:
        etf = fetch_510300_daily(pad_start, pad_end)
    else:
        raise ValueError(f"unsupported market={market}")
    daily_ret = daily_returns(etf.set_index("date")["close"])
    return forward_holding_returns(daily_ret, formation_dates, HORIZON)


def vol_adjusted_cum_log(
    strategy_returns: pd.Series,
    benchmark_vol: float,
) -> pd.Series:
    scaled = scale_to_benchmark_vol(strategy_returns, benchmark_vol)
    return cum_log_returns(scaled)


def plot_font(size: float) -> font_manager.FontProperties:
    """Times New Roman when installed; otherwise Nimbus Roman (URW Times clone)."""
    tnr = font_manager.FontProperties(family=FONT_FAMILY, size=size)
    tnr_path = font_manager.findfont(tnr)
    if "dejavu" not in tnr_path.lower():
        return tnr
    nimbus = font_manager.FontProperties(fname=NIMBUS_ROMAN_PATH, size=size)
    if not Path(NIMBUS_ROMAN_PATH).is_file():
        raise FileNotFoundError(f"missing Times substitute font: {NIMBUS_ROMAN_PATH}")
    font_manager.fontManager.addfont(NIMBUS_ROMAN_PATH)
    return nimbus


def _apply_font(ax: plt.Axes, legend: plt.Legend, *, size: float) -> None:
    fp = plot_font(size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp)
    for text in legend.get_texts():
        text.set_fontproperties(fp)


def plot_cumulative(
    curves: dict[str, pd.Series],
    *,
    market: str,
    vol_adjusted: bool,
    out_path: Path,
) -> None:
    fig_w, fig_h = FIG_W, FIG_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes(
        [
            LEFT_IN / fig_w,
            BOTTOM_IN / fig_h,
            PLOT_IN / fig_w,
            PLOT_IN / fig_h,
        ]
    )

    for label, series in curves.items():
        style = series_style(label)
        ax.plot(
            series.index,
            series.values,
            label=label,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
        )

    tick_size = 5.5
    ax.tick_params(axis="both", labelsize=tick_size)
    legend = ax.legend(
        loc="upper left",
        prop=plot_font(tick_size),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
        ncol=2,
        columnspacing=0.8,
        handlelength=1.2,
        borderpad=0.35,
        labelspacing=0.25,
    )
    _apply_font(ax, legend, size=tick_size)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_clip_on(False)

    xs = [t for s in curves.values() for t in s.index]
    xmin, xmax = min(xs), max(xs)
    xpad = (xmax - xmin) * 0.01
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.autoscale(enable=True, axis="y", tight=True)
    ax.margins(x=0, y=0.02)

    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    log(f"wrote {out_path}")


def run_market(market: str) -> list[Path]:
    hl = load_all_hl(market)
    etf_label = BENCHMARK_ETF[market][0]
    ref_dates = hl["I5/R5"].index
    etf_holding = load_benchmark_etf_holding_returns(market, ref_dates)
    hl[etf_label] = etf_holding
    bench_vol = period_volatility(etf_holding)

    raw_curves = {label: cum_log_returns(s) for label, s in hl.items()}
    vol_curves: dict[str, pd.Series] = {}
    for label, s in hl.items():
        if label == etf_label:
            vol_curves[label] = cum_log_returns(s)
        else:
            vol_curves[label] = vol_adjusted_cum_log(s, bench_vol)

    raw_path = HERE / f"{STEM}_raw_{market}.png"
    vol_path = HERE / f"{STEM}_voladj_{market}.png"
    written: list[Path] = []
    plot_cumulative(raw_curves, market=market, vol_adjusted=False, out_path=raw_path)
    plot_cumulative(vol_curves, market=market, vol_adjusted=True, out_path=vol_path)
    written.extend([raw_path, vol_path])
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 5 cumulative H–L return plots")
    parser.add_argument(
        "--market",
        choices=(MARKET_US, MARKET_CN),
        default=None,
        help="single market (default: both us and cn)",
    )
    args = parser.parse_args()
    markets = [args.market] if args.market else [MARKET_US, MARKET_CN]
    for market in markets:
        run_market(market)


if __name__ == "__main__":
    main()
