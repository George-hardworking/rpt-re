"""Figure-8 style Sharpe-ratio distribution plots for STW rules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE8_HORIZONS: tuple[int, ...] = (5, 20, 60)
FIGURE8_PANEL_LABELS: tuple[str, ...] = ("I20/R5", "I20/R20", "I20/R60")
FIGURE8_ROW_SCHEMES: tuple[tuple[str, str], ...] = (
    ("equal", "Equal weight"),
    ("total", "Value weight"),
)
TIMES_FONT = "Times New Roman"


def _use_times_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [TIMES_FONT, "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
        }
    )


def _clean_sharpes(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def _scheme_sharpes(sharpe_df: pd.DataFrame, scheme: str) -> np.ndarray:
    mask = sharpe_df["weight_scheme"] == scheme
    return _clean_sharpes(sharpe_df.loc[mask, "sharpe"])


def _panel_xlim(vals: np.ndarray, cnn_sharpe: float | None, *, pad_frac: float = 0.05) -> tuple[float, float]:
    """X limits that include both the histogram and the CNN reference line."""
    if len(vals) == 0:
        if cnn_sharpe is not None and np.isfinite(cnn_sharpe):
            return (cnn_sharpe - 0.5, cnn_sharpe + 0.5)
        return (-1.0, 1.0)

    x_lo = float(np.min(vals))
    x_hi = float(np.max(vals))
    if cnn_sharpe is not None and np.isfinite(cnn_sharpe):
        x_lo = min(x_lo, cnn_sharpe)
        x_hi = max(x_hi, cnn_sharpe)
    span = x_hi - x_lo
    pad = pad_frac * span if span > 0 else 0.5
    return (x_lo - pad, x_hi + pad)


def _style_panel_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)


def _draw_histogram_panel(
    ax: plt.Axes,
    vals: np.ndarray,
    *,
    cnn_sharpe: float | None,
    bins: int,
) -> None:
    ax.hist(vals, bins=bins, color="#4472C4", edgecolor="white", linewidth=0.35)
    if cnn_sharpe is not None and np.isfinite(cnn_sharpe):
        ax.axvline(cnn_sharpe, color="#c92228", linewidth=2.0)
    _style_panel_ax(ax)


def plot_stw_figure8_panel(
    panels: dict[int, pd.DataFrame],
    cnn_sharpes: dict[int, dict[str, float]],
    *,
    output_path: Path,
    market: str,
    bins: int = 80,
) -> Path:
    """2×3 Figure-8 panel: equal row + value-weight row × I20/R5,R20,R60."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _use_times_font()

    missing = [h for h in FIGURE8_HORIZONS if h not in panels]
    if missing:
        raise KeyError(f"missing STW sharpe panels for horizons={missing}")

    # Width slightly > height; 2×3 grid => each panel is wider than tall when w/h > 1.5.
    fig_w, fig_h = 10.0, 6.5

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(fig_w, fig_h),
        facecolor="white",
    )
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(1.0)

    for row, (scheme, row_label) in enumerate(FIGURE8_ROW_SCHEMES):
        for col, horizon in enumerate(FIGURE8_HORIZONS):
            ax = axes[row, col]
            vals = _scheme_sharpes(panels[horizon], scheme)
            cnn = cnn_sharpes.get(horizon, {}).get(scheme)
            _draw_histogram_panel(ax, vals, cnn_sharpe=cnn, bins=bins)
            ax.set_xlim(_panel_xlim(vals, cnn))
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.set_title(FIGURE8_PANEL_LABELS[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nfrequency", fontsize=9)
            else:
                ax.set_ylabel("frequency", fontsize=9)

    fig.supxlabel("Annualized H-L Sharpe ratio", fontsize=10, y=0.03)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.90, wspace=0.28, hspace=0.32)
    fig.savefig(
        output_path,
        dpi=220,
        facecolor="white",
        edgecolor="black",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    return output_path


def plot_stw_sharpe_distribution(
    sharpe_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    cnn_sharpes: dict[str, float] | None = None,
    bins: int = 80,
    schemes: list[str] | None = None,
) -> Path:
    """Single-panel histogram (legacy helper)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cnn_sharpes = cnn_sharpes or {}
    _use_times_font()

    available = [s for s in ("equal", "float", "total") if s in set(sharpe_df["weight_scheme"])]
    plot_schemes = schemes if schemes is not None else available
    plot_schemes = [s for s in plot_schemes if s in available]
    if not plot_schemes:
        raise ValueError("no supported weight_scheme values found for STW plot")

    fig, axes = plt.subplots(
        len(plot_schemes),
        1,
        figsize=(8.0, 2.8 * len(plot_schemes)),
        sharex=True,
        facecolor="white",
        constrained_layout=True,
    )
    if len(plot_schemes) == 1:
        axes = [axes]

    for ax, scheme in zip(axes, plot_schemes):
        vals = _scheme_sharpes(sharpe_df, scheme)
        cnn = cnn_sharpes.get(scheme)
        _draw_histogram_panel(ax, vals, cnn_sharpe=cnn, bins=bins)
        ax.set_xlim(_panel_xlim(vals, cnn))
        ax.set_ylabel(f"{scheme}\nfrequency")

    axes[-1].set_xlabel("Annualized H-L Sharpe ratio", fontsize=10)
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return output_path
