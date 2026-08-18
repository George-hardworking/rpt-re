"""Figure-8 style Sharpe-ratio distribution plots for STW rules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _clean_sharpes(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def plot_stw_sharpe_distribution(
    sharpe_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
    cnn_sharpes: dict[str, float] | None = None,
    bins: int = 80,
    schemes: list[str] | None = None,
) -> Path:
    """Write a paper-style histogram of STW rule Sharpes with CNN red lines.

    The input frame is the combined output from `08_stw_7846_rules.py backtest`,
    with columns `weight_scheme` and `sharpe`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cnn_sharpes = cnn_sharpes or {}

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
        constrained_layout=True,
    )
    if len(plot_schemes) == 1:
        axes = [axes]

    for ax, scheme in zip(axes, plot_schemes):
        vals = _clean_sharpes(sharpe_df.loc[sharpe_df["weight_scheme"] == scheme, "sharpe"])
        ax.hist(vals, bins=bins, color="#6f7885", edgecolor="white", linewidth=0.35)
        cnn = cnn_sharpes.get(scheme)
        if cnn is not None and np.isfinite(cnn):
            ax.axvline(cnn, color="#c92228", linewidth=2.2, label=f"CNN Sharpe = {cnn:.3f}")
            ax.legend(frameon=False, loc="upper right")
        ax.set_ylabel(f"{scheme}\ncount")
        ax.grid(axis="y", color="#d7dce2", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Annualized H-L Sharpe ratio")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
