"""04 — transfer decile monotonicity (CN local vs US direct vs US finetune).

Read-only: loads step-04 CNN baseline and step-07 transfer all_h1.xlsx (equal-weight).
Nine series per frequency: 3 sources × 3 image sizes (I5/I20/I60 at R5/R20/R60).
Markers: CN local circle, US direct triangle, US finetune square.
Single panel: equal-weight annualized return by decile. Legend sits below the axes.

Run (from repo root, 5020_env):
  python "outputs/analysis_results/04_transfer_decile_curves/04_transfer_decile_figure.py"
  python "outputs/analysis_results/04_transfer_decile_curves/04_transfer_decile_figure.py" --freq monthly
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from config import (
    BACKTEST_CNN_ROOT,
    TRANSFER_BACKTEST_DIRECT_ROOT,
    TRANSFER_BACKTEST_FINETUNE_ROOT,
)

HERE = Path(__file__).resolve().parent
STEM = "transfer_decile_curves"

RET_METRIC = "Annualized Return"
DECILE_LABELS = tuple(f"D{i:02d}" for i in range(1, 11))

# Single-panel layout (return only); extra bottom margin for legend below axes.
PLOT_W = 2.30
PLOT_H = 1.82
LEFT_IN = 0.50
RIGHT_IN = 0.04
BOTTOM_IN = 0.62
TOP_IN = 0.04
FIG_W = LEFT_IN + PLOT_W + RIGHT_IN
FIG_H = BOTTOM_IN + PLOT_H + TOP_IN


@dataclass(frozen=True)
class FreqSpec:
    name: str
    horizon: int


FREQ_SPECS: dict[str, FreqSpec] = {
    "weekly": FreqSpec("weekly", 5),
    "monthly": FreqSpec("monthly", 20),
    "quarterly": FreqSpec("quarterly", 60),
}

TRANSFER_SOURCES: tuple[tuple[str, Path, str, str, str], ...] = (
    ("cn_local", BACKTEST_CNN_ROOT / "cn", "CN local", "-", "o"),
    ("us_direct", TRANSFER_BACKTEST_DIRECT_ROOT / "cn", "US direct", "--", "^"),
    ("us_finetune", TRANSFER_BACKTEST_FINETUNE_ROOT / "cn", "US finetune", "-.", "s"),
)

IMAGE_COLORS: dict[int, str] = {
    5: "#d62728",
    20: "#1f77b4",
    60: "#ff7f0e",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _load_decile_style_module():
    path = ROOT / "outputs/analysis_results/03_decile_scatter_curves/03_decile_figure.py"
    spec = importlib.util.spec_from_file_location("decile_figure_style", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def read_h1_models(path: Path, sheet: str = "equal") -> dict[str, pd.Series]:
    portfolio_mod = _load_portfolio_module()
    return portfolio_mod.read_h1_models(path, sheet)


def _decile_values(row: pd.Series, metric: str) -> np.ndarray:
    return np.array([float(row[(metric, d)]) for d in DECILE_LABELS], dtype=np.float64)


def strategies_for_freq(spec: FreqSpec) -> tuple[tuple[str, str, str, str, str], ...]:
    out: list[tuple[str, str, str, str, str]] = []
    for _source_id, root, source_label, linestyle, marker in TRANSFER_SOURCES:
        for image_days in (5, 20, 60):
            col = f"{source_label} I{image_days}"
            color = IMAGE_COLORS[image_days]
            out.append((col, col, linestyle, color, marker))
    return tuple(out)


def transfer_all_path(source_root: Path, freq: str) -> Path:
    return source_root / freq / "all_h1.xlsx"


def load_transfer_panel(
    spec: FreqSpec,
) -> tuple[dict[str, np.ndarray], tuple[tuple[str, str, str, str, str], ...]]:
    strategies = strategies_for_freq(spec)
    series: dict[str, np.ndarray] = {}
    for source_id, root, source_label, _ls, _marker in TRANSFER_SOURCES:
        path = transfer_all_path(root, spec.name)
        models = read_h1_models(path, sheet="equal")
        for image_days in (5, 20, 60):
            row_name = f"I{image_days}_R{spec.horizon}"
            if row_name not in models:
                raise KeyError(f"{path} missing row {row_name!r} for source={source_id}")
            row = models[row_name]
            col = f"{source_label} I{image_days}"
            series[col] = _decile_values(row, RET_METRIC)
    return series, strategies


def plot_transfer_decile_curves(
    series: dict[str, np.ndarray],
    strategies: tuple[tuple[str, str, str, str, str], ...],
    *,
    freq: str,
    out_path: Path,
) -> None:
    style = _load_decile_style_module()
    ret_flat = np.concatenate(list(series.values()))
    ret_y0, ret_y1 = style._data_limits(ret_flat)

    fig_w, fig_h = FIG_W, FIG_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([LEFT_IN / fig_w, BOTTOM_IN / fig_h, PLOT_W / fig_w, PLOT_H / fig_h])
    tick_size = 7.0

    style._apply_ticks(ax, ret_y0, ret_y1, style.RET_STEP, tick_size=tick_size)
    ax.set_ylabel(
        "Annualized Return",
        fontproperties=style.plot_font(tick_size + 0.5),
        labelpad=3.0,
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(style.SPINE_LW)
        ax.spines[side].set_clip_on(True)

    for col, legend, ls, color, marker in strategies:
        ret = series[col]
        ax.plot(
            style.DECILE_X,
            ret,
            linestyle=ls,
            color=color,
            linewidth=0.9 if ls == "-" else 0.75,
            marker=marker,
            markersize=4.2 if marker == "^" else 4.0,
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=0.8,
            clip_on=False,
            label=legend,
        )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        prop=style.plot_font(tick_size - 1.0),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
        handlelength=1.4,
        borderpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
    )
    for text in legend.get_texts():
        text.set_fontproperties(style.plot_font(tick_size - 1.0))

    fig.savefig(
        out_path,
        dpi=style.FIG_DPI,
        bbox_inches="tight",
        pad_inches=style.SAVE_PAD_INCHES,
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    log(f"wrote {out_path} ({freq})")


def output_path(freq: str) -> Path:
    return HERE / f"{STEM}_{freq}_cn.png"


def run_freq(freq: str) -> Path:
    spec = FREQ_SPECS[freq]
    series, strategies = load_transfer_panel(spec)
    out_path = output_path(freq)
    plot_transfer_decile_curves(series, strategies, freq=freq, out_path=out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer decile return monotonicity (CN)")
    parser.add_argument(
        "--freq",
        choices=tuple(FREQ_SPECS),
        default=None,
        help="single rebalance frequency (default: weekly, monthly, and quarterly)",
    )
    args = parser.parse_args()
    freqs = tuple(FREQ_SPECS) if args.freq is None else (args.freq,)
    for freq in freqs:
        run_freq(freq)


if __name__ == "__main__":
    main()
