"""03 — decile portfolio curves: annualized return & risk (I5 + benchmarks).

Read-only: loads step-04 all_h1.xlsx and step-05 h1.xlsx (equal-weight, weekly/monthly/quarterly).
Five series: I5 (solid red), MOM / two reversals / TREND (dashed). Hollow circle markers.
Two panels side-by-side; y-axis ticks at 0.1 (return) and 0.05 (risk); axis limits use
exact data extrema so the envelope touches the plot bottom-left and top-right corners.

Run (from repo root, 5020_env):
  python "outputs/analysis_results/03_decile_scatter_curves/03_decile_figure.py"
  python "outputs/analysis_results/03_decile_scatter_curves/03_decile_figure.py" --freq monthly --market cn
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from config import MARKET_CN, MARKET_US

HERE = Path(__file__).resolve().parent
STEM = "decile_curves"
# Plot box: y a bit longer than x, but not a square.
PLOT_W = 1.60
PLOT_H = 1.82
LEFT_IN = 0.50
GAP_IN = 0.60
RIGHT_IN = 0.04
BOTTOM_IN = 0.22
TOP_IN = 0.04
FIG_W = LEFT_IN + PLOT_W + GAP_IN + PLOT_W + RIGHT_IN
FIG_H = BOTTOM_IN + PLOT_H + TOP_IN
FIG_DPI = 300
# Match 02 cumulative-return plot boxes (pt).
SPINE_LW = 0.25
SAVE_PAD_INCHES = SPINE_LW / 72.0 / 2.0 + 0.002
FONT_FAMILY = "Times New Roman"
NIMBUS_ROMAN_PATH = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"

RET_METRIC = "Annualized Return"
VOL_METRIC = "Annualized Risk"
RET_STEP = 0.1
VOL_STEP = 0.05
DECILE_X = np.arange(1, 11, dtype=np.float64)
DECILE_LABELS = tuple(f"D{i:02d}" for i in range(1, 11))
DECILE_TICK_LABELS = tuple(str(i) for i in range(1, 11))

def log(msg: str) -> None:
    print(msg, flush=True)


def strategies_for_horizon(horizon: int) -> tuple[tuple[str, str, str, str], ...]:
    h = horizon
    return (
        (f"I5/R{h}", "I5", "-", "#d62728"),
        (f"MOM/R{h}", "MOM", "--", "#2ca02c"),
        (f"STR/R{h}", "REV1m", "--", "#9467bd"),
        (f"WSTR/R{h}", "REV1w", "--", "#17becf"),
        (f"TREND/R{h}", "TREND", "--", "#8b0000"),
    )


def _load_portfolio_module():
    path = (
        ROOT
        / "outputs/analysis_results/01_short-horizon portfolio performance"
        / "01_portfolio_weekly_h1.py"
    )
    spec = importlib.util.spec_from_file_location("portfolio_h1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def plot_font(size: float) -> font_manager.FontProperties:
    tnr = font_manager.FontProperties(family=FONT_FAMILY, size=size)
    tnr_path = font_manager.findfont(tnr)
    if "dejavu" not in tnr_path.lower():
        return tnr
    nimbus = font_manager.FontProperties(fname=NIMBUS_ROMAN_PATH, size=size)
    if not Path(NIMBUS_ROMAN_PATH).is_file():
        raise FileNotFoundError(f"missing Times substitute font: {NIMBUS_ROMAN_PATH}")
    font_manager.fontManager.addfont(NIMBUS_ROMAN_PATH)
    return nimbus


def _data_limits(values: np.ndarray) -> tuple[float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        hi = lo + 1e-6
    return lo, hi


def _y_tick_values(y0: float, y1: float, step: float) -> np.ndarray:
    tick_lo = np.floor(y0 / step) * step
    tick_hi = np.ceil(y1 / step) * step
    return np.arange(tick_lo, tick_hi + step * 0.5, step)


def _apply_ticks(ax: plt.Axes, y0: float, y1: float, step: float, *, tick_size: float) -> None:
    ax.set_yticks(_y_tick_values(y0, y1, step))
    ax.set_xticks(DECILE_X)
    ax.set_xticklabels(DECILE_TICK_LABELS)
    ax.set_xlim(1.0, 10.0)
    ax.set_ylim(y0, y1)
    ax.margins(x=0, y=0)
    ax.autoscale(enable=False)
    ax.tick_params(axis="both", labelsize=tick_size, width=SPINE_LW, length=2.2, pad=1.5)
    fp = plot_font(tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(fp)


def _decile_values(row, metric: str) -> np.ndarray:
    return np.array([float(row[(metric, d)]) for d in DECILE_LABELS], dtype=np.float64)


def load_decile_panel(
    portfolio_mod,
    market: str,
    freq: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], tuple[tuple[str, str, str, str], ...]]:
    spec = portfolio_mod.FREQ_SPECS[freq]
    strategies = strategies_for_horizon(spec.horizon)
    panel = portfolio_mod.load_panel(spec, market, sheet="equal", eval_horizon=1)
    columns = [s[0] for s in strategies]
    missing = [c for c in columns if c not in panel]
    if missing:
        raise KeyError(f"missing strategies in panel ({freq} {market}): {missing}")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in columns:
        row = panel[col]
        out[col] = (_decile_values(row, RET_METRIC), _decile_values(row, VOL_METRIC))
    return out, strategies


def plot_decile_curves(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    strategies: tuple[tuple[str, str, str, str], ...],
    *,
    market: str,
    out_path: Path,
) -> None:
    all_ret = [v[0] for v in series.values()]
    all_vol = [v[1] for v in series.values()]
    ret_flat = np.concatenate(all_ret)
    vol_flat = np.concatenate(all_vol)
    ret_y0, ret_y1 = _data_limits(ret_flat)
    vol_y0, vol_y1 = _data_limits(vol_flat)

    fig_w, fig_h = FIG_W, FIG_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_left = fig.add_axes(
        [LEFT_IN / fig_w, BOTTOM_IN / fig_h, PLOT_W / fig_w, PLOT_H / fig_h]
    )
    ax_right = fig.add_axes(
        [
            (LEFT_IN + PLOT_W + GAP_IN) / fig_w,
            BOTTOM_IN / fig_h,
            PLOT_W / fig_w,
            PLOT_H / fig_h,
        ]
    )
    ax_right.sharex(ax_left)
    axes = (ax_left, ax_right)
    tick_size = 7.0
    ylabels = ("Annualized Return", "Annualized Volatility")

    for ax, ylabel, (y0, y1, step) in zip(
        axes,
        ylabels,
        ((ret_y0, ret_y1, RET_STEP), (vol_y0, vol_y1, VOL_STEP)),
    ):
        _apply_ticks(ax, y0, y1, step, tick_size=tick_size)
        ax.set_ylabel(ylabel, fontproperties=plot_font(tick_size + 0.5), labelpad=3.0)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
        for side in ("left", "right", "top", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(SPINE_LW)
            ax.spines[side].set_clip_on(True)

    for col, legend, ls, color in strategies:
        ret, vol = series[col]
        for ax, ys in zip(axes, (ret, vol)):
            ax.plot(
                DECILE_X,
                ys,
                linestyle=ls,
                color=color,
                linewidth=0.9 if ls == "-" else 0.75,
                marker="o",
                markersize=4.0,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=0.8,
                clip_on=False,
                label=legend,
            )

    legend = axes[0].legend(
        loc="upper left",
        prop=plot_font(tick_size - 0.5),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.3,
    )
    for text in legend.get_texts():
        text.set_fontproperties(plot_font(tick_size - 0.5))

    fig.savefig(
        out_path,
        dpi=FIG_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    log(f"wrote {out_path}")


def output_path(freq: str, market: str) -> Path:
    return HERE / f"{STEM}_{freq}_{market}.png"


def run_market(market: str, freq: str) -> Path:
    portfolio_mod = _load_portfolio_module()
    series, strategies = load_decile_panel(portfolio_mod, market, freq)
    out_path = output_path(freq, market)
    plot_decile_curves(series, strategies, market=market, out_path=out_path)
    return out_path


def main() -> None:
    portfolio_mod = _load_portfolio_module()
    freq_choices = tuple(portfolio_mod.FREQ_SPECS)
    parser = argparse.ArgumentParser(description="Decile return & volatility scatter curves")
    parser.add_argument(
        "--market",
        choices=(MARKET_US, MARKET_CN),
        default=None,
        help="single market (default: both us and cn)",
    )
    parser.add_argument(
        "--freq",
        choices=freq_choices,
        default=None,
        help="single rebalance frequency (default: weekly, monthly, and quarterly)",
    )
    args = parser.parse_args()
    markets = [args.market] if args.market else [MARKET_US, MARKET_CN]
    freqs = freq_choices if args.freq is None else (args.freq,)
    for freq in freqs:
        for market in markets:
            run_market(market, freq)


if __name__ == "__main__":
    main()
