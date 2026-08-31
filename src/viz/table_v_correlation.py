"""Table V heatmap: mean rank correlations of CNN forecasts vs characteristics."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.table_format import format_value_stars

CORR_VMIN = -1.0
CORR_VMAX = 1.0
TIMES_FONT = "Times New Roman"
TIMES_FONT_DIR = Path.home() / ".local" / "share" / "fonts"
MARKET_TITLES = {"us": "US", "cn": "CN"}


def _use_times_font() -> None:
    for name in (
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf",
        "Times New Roman Italic.ttf",
        "Times New Roman Bold Italic.ttf",
    ):
        path = TIMES_FONT_DIR / name
        assert path.is_file(), f"missing Times New Roman at {path}"
        mpl.font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": TIMES_FONT,
            "font.serif": [TIMES_FONT],
            "mathtext.fontset": "stix",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
        }
    )


def _cell_text_color(value: float, cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize) -> str:
    r, g, b, _ = cmap(norm(value))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.55 else "black"


def plot_table_v_heatmap(
    mean_vals: pd.DataFrame,
    t_stats: pd.DataFrame,
    *,
    market: str,
    output_path: Path,
) -> Path:
    assert mean_vals.shape == t_stats.shape
    assert list(mean_vals.index) == list(t_stats.index)
    assert list(mean_vals.columns) == list(t_stats.columns)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _use_times_font()

    values = mean_vals.to_numpy(dtype=np.float64)
    tmat = t_stats.to_numpy(dtype=np.float64)
    n_rows, n_cols = values.shape
    cmap = mpl.colormaps["RdBu_r"]
    norm = mpl.colors.TwoSlopeNorm(vmin=CORR_VMIN, vcenter=0.0, vmax=CORR_VMAX)

    fig_w = 1.05 + 0.92 * n_cols
    fig_h = 1.35 + 0.42 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(
        list(mean_vals.columns),
        rotation=40,
        ha="right",
        fontsize=9,
        fontfamily=TIMES_FONT,
    )
    ax.set_yticklabels(list(mean_vals.index), fontsize=9, fontfamily=TIMES_FONT)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            ax.text(
                j,
                i,
                format_value_stars(val, tmat[i, j]),
                ha="center",
                va="center",
                fontsize=7.2,
                fontfamily=TIMES_FONT,
                color=_cell_text_color(val, cmap, norm),
            )

    market_label = MARKET_TITLES[market]
    ax.set_title(
        f"{market_label}: CNN forecasts vs characteristics",
        fontsize=11,
        pad=8,
        fontfamily=TIMES_FONT,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean rank correlation", fontsize=9, fontfamily=TIMES_FONT)
    cbar.set_ticks(np.linspace(CORR_VMIN, CORR_VMAX, 5))
    cbar.ax.tick_params(labelsize=8)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(TIMES_FONT)

    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output_path
