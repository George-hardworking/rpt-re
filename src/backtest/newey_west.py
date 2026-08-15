"""Newey-West lag selection, mean t-stat, and significance formatting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NW_LAGS_AUTO = "auto"


def newey_west_lags_from_periods(n_periods: int, *, min_lags: int = 1) -> int:
    """L = floor(4 * (T / 100) ** (2/9)); T = number of time periods."""
    t = int(n_periods)
    if t < 2:
        return 0
    lags = int(4 * (t / 100) ** (2 / 9))
    lags = max(int(min_lags), lags)
    return min(lags, t - 1)


def _is_auto(spec: Any) -> bool:
    if spec is None:
        return True
    if isinstance(spec, str) and spec.strip().lower() in {"auto", "null", "none", ""}:
        return True
    return False


def resolve_newey_west_lags(n_periods: int, lags_spec: Any) -> int:
    """0 → plain SE; positive int → fixed lags; None / \"auto\" → empirical rule."""
    if isinstance(lags_spec, (int, float)) and not isinstance(lags_spec, bool):
        lags = int(lags_spec)
        if lags <= 0:
            return 0
        t = int(n_periods)
        return min(lags, max(t - 1, 1)) if t >= 2 else 0
    if _is_auto(lags_spec):
        return newey_west_lags_from_periods(n_periods)
    if isinstance(lags_spec, str) and lags_spec.strip().isdigit():
        return resolve_newey_west_lags(n_periods, int(lags_spec.strip()))
    raise ValueError(f"Invalid newey_west_lags spec: {lags_spec!r}")


def newey_west_se(series: np.ndarray, lags: int) -> float:
    """NW standard error of the sample mean (univariate)."""
    arr = np.asarray(series, dtype=np.float64).ravel()
    n = len(arr)
    if n < 2:
        return float("nan")
    mean = float(arr.mean())
    errors = arr - mean
    gamma0 = float(np.dot(errors, errors) / n)
    nw_var = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma_l = float(np.dot(errors[lag:], errors[:-lag]) / n)
        nw_var += 2.0 * weight * gamma_l
    return float(np.sqrt(nw_var / n))


def nw_mean_tstat(
    series: pd.Series | np.ndarray,
    *,
    newey_west_lags: Any = NW_LAGS_AUTO,
) -> tuple[float, float]:
    """Newey-West t-stat for H0: mean(series) = 0."""
    arr = pd.Series(series).dropna().astype(float).to_numpy()
    n = len(arr)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    resolved = resolve_newey_west_lags(n, newey_west_lags)
    if resolved > 0:
        se = newey_west_se(arr, resolved)
    else:
        se = float(arr.std(ddof=1) / np.sqrt(n))
    if not np.isfinite(se) or se <= 0.0:
        return mean, float("nan")
    return mean, float(mean / se)


def significance_stars(t_stat: float) -> str:
    abs_t = abs(float(t_stat))
    if abs_t >= 2.576:
        return "***"
    if abs_t >= 1.960:
        return "**"
    if abs_t >= 1.645:
        return "*"
    return ""


def format_nw_t(t_stat: float, *, decimals: int = 2) -> str:
    """Single-cell display: e.g. ``5.85***``."""
    if not np.isfinite(t_stat):
        return ""
    return f"{t_stat:.{decimals}f}{significance_stars(t_stat)}"
