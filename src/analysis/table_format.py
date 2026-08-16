"""Compact display cells: value + significance stars + (t-stat)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.newey_west import significance_stars


def format_cell(
    value: float,
    t_stat: float,
    *,
    value_decimals: int = 2,
    t_decimals: int = 1,
) -> str:
    """Format ``-0.34*** (-2.8)``; empty string when value is not finite."""
    if not np.isfinite(value):
        return ""
    stars = significance_stars(t_stat) if np.isfinite(t_stat) else ""
    t_part = f"({t_stat:.{t_decimals}f})" if np.isfinite(t_stat) else ""
    body = f"{value:.{value_decimals}f}{stars}"
    return f"{body} {t_part}".strip()


def format_value_only(value: float, *, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.{decimals}f}"


def build_display_matrix(
    values: pd.DataFrame,
    t_stats: pd.DataFrame,
    *,
    value_decimals: int = 2,
    t_decimals: int = 1,
) -> pd.DataFrame:
    """Element-wise ``format_cell`` for aligned value/t frames."""
    assert values.shape == t_stats.shape
    out = values.copy()
    for row in values.index:
        for col in values.columns:
            out.at[row, col] = format_cell(
                values.at[row, col],
                t_stats.at[row, col],
                value_decimals=value_decimals,
                t_decimals=t_decimals,
            )
    return out


def write_display_raw_excel(
    path: Path,
    *,
    display: pd.DataFrame,
    raw_values: pd.DataFrame,
    raw_t: pd.DataFrame,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.xlsx")
    raw = raw_values.copy()
    raw.columns = [f"{c}_val" for c in raw.columns]
    for col in raw_t.columns:
        raw[f"{col}_t"] = raw_t[col]
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        display.to_excel(writer, sheet_name="display")
        raw.to_excel(writer, sheet_name="raw")
    tmp.replace(path)
    return path
