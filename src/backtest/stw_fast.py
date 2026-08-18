"""Fast H-L Sharpe computation for STW rule chunks.

This module intentionally computes only the distribution needed for the
Figure-8 style comparison: annualized Sharpe ratios of high-minus-low
decile spreads.  It avoids pandas rank/groupby work inside the rule loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.stw_signals import MISSING_SIGNAL


@dataclass(frozen=True)
class STWSharpeConfig:
    date_col: str
    ret_col: str
    periods_per_year: int
    ngroup: int = 10


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(values * weights) / denom)


def _repo_standardize_one_date(signal: np.ndarray) -> np.ndarray:
    """Match backtest.engine.standardize_by_date for one date and one signal."""
    x = signal.astype(np.float64, copy=True)
    x[~np.isfinite(x)] = np.nan
    mean = np.nanmean(x)
    std = np.nanstd(x, ddof=1)
    if not np.isfinite(std) or std <= 0.0:
        return np.full(len(x), np.nan, dtype=np.float64)
    z = (x - mean) / std
    z = np.clip(z, -3.0, 3.0)
    mean2 = np.nanmean(z)
    std2 = np.nanstd(z, ddof=1)
    if not np.isfinite(std2) or std2 <= 0.0:
        return np.full(len(x), np.nan, dtype=np.float64)
    return (z - mean2) / std2


def _repo_groups_one_date(signal: np.ndarray, ngroup: int) -> np.ndarray:
    """Match assign_quantile_groups(rank(method='first')) for one date."""
    n = len(signal)
    groups = np.zeros(n, dtype=np.int16)
    ok = np.isfinite(signal)
    n_ok = int(ok.sum())
    if n_ok < ngroup:
        return groups
    order = np.argsort(signal[ok], kind="mergesort")
    ranks = np.empty(n_ok, dtype=np.float64)
    ranks[order] = np.arange(1, n_ok + 1, dtype=np.float64)
    group_ok = np.floor((ranks - 1.0) / n_ok * ngroup).astype(np.int16) + 1
    group_ok = np.clip(group_ok, 1, ngroup)
    groups[np.flatnonzero(ok)] = group_ok
    return groups


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> tuple[float, float, float, int]:
    x = returns[np.isfinite(returns)]
    n = int(len(x))
    if n < 2:
        return np.nan, np.nan, np.nan, n
    mean = float(np.mean(x) * periods_per_year)
    risk = float(np.std(x, ddof=1) * np.sqrt(periods_per_year))
    sharpe = mean / risk if risk > 0.0 else np.nan
    return mean, risk, sharpe, n


def high_low_returns_matrix(
    panel: pd.DataFrame,
    *,
    signal_cols: list[str],
    config: STWSharpeConfig,
    weight_col: str | None,
) -> pd.DataFrame:
    """Return date x rule matrix of H-L decile spread returns."""
    need = [config.date_col, config.ret_col, *signal_cols]
    if weight_col is not None:
        need.append(weight_col)
    missing = [c for c in need if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing columns {missing}")

    work = panel[need].copy()
    work[config.date_col] = pd.to_datetime(work[config.date_col])
    work = work.sort_values(config.date_col, kind="mergesort").reset_index(drop=True)

    signals = work[signal_cols].to_numpy(dtype=np.float32, copy=False)
    returns = work[config.ret_col].to_numpy(dtype=np.float64, copy=False)
    if weight_col is None:
        weights = np.ones(len(work), dtype=np.float64)
    else:
        weights = work[weight_col].to_numpy(dtype=np.float64, copy=False)

    dates = pd.DatetimeIndex(pd.unique(work[config.date_col]))
    groups = work.groupby(config.date_col, sort=False).indices
    hl = np.full((len(dates), len(signal_cols)), np.nan, dtype=np.float32)

    for t, date in enumerate(dates):
        idx = np.asarray(groups[date], dtype=np.int64)
        ret_t = returns[idx]
        w_t = weights[idx]
        ret_ok = np.isfinite(ret_t)
        if int(ret_ok.sum()) < config.ngroup:
            continue

        sig_t = signals[idx, :]
        for j in range(sig_t.shape[1]):
            s = sig_t[:, j]
            rank_ok = ret_ok & np.isfinite(s) & (s != float(MISSING_SIGNAL))
            if int(rank_ok.sum()) < config.ngroup:
                continue
            standardized = np.full(len(s), np.nan, dtype=np.float64)
            standardized[rank_ok] = _repo_standardize_one_date(s[rank_ok])
            group = _repo_groups_one_date(standardized, config.ngroup)

            portfolio_ok = rank_ok & np.isfinite(w_t) & (w_t > 0.0)
            low = portfolio_ok & (group == 1)
            high = portfolio_ok & (group == config.ngroup)
            if not low.any() or not high.any():
                continue
            r_low = _weighted_mean(ret_t[low], w_t[low])
            r_high = _weighted_mean(ret_t[high], w_t[high])
            if np.isfinite(r_low) and np.isfinite(r_high):
                hl[t, j] = np.float32(r_high - r_low)

    return pd.DataFrame(hl, index=dates, columns=signal_cols)


def summarize_hl_sharpes(
    hl_returns: pd.DataFrame,
    *,
    periods_per_year: int,
    weight_scheme: str,
) -> pd.DataFrame:
    rows = []
    for col in hl_returns.columns:
        mean, risk, sharpe, n = _annualized_sharpe(
            hl_returns[col].to_numpy(dtype=np.float64),
            periods_per_year,
        )
        rows.append(
            {
                "rule_name": col,
                "weight_scheme": weight_scheme,
                "n_periods": n,
                "annualized_return": mean,
                "annualized_risk": risk,
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows)


def fast_rule_chunk_sharpes(
    panel: pd.DataFrame,
    *,
    signal_cols: list[str],
    config: STWSharpeConfig,
    weight_schemes: dict[str, str | None],
) -> pd.DataFrame:
    """Compute H-L Sharpe rows for one rule-column chunk."""
    out = []
    for scheme, weight_col in weight_schemes.items():
        hl = high_low_returns_matrix(
            panel,
            signal_cols=signal_cols,
            config=config,
            weight_col=weight_col,
        )
        out.append(
            summarize_hl_sharpes(
                hl,
                periods_per_year=config.periods_per_year,
                weight_scheme=scheme,
            )
        )
    return pd.concat(out, ignore_index=True)
