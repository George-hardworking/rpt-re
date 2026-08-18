"""Benchmark ETF daily prices via akshare (SPY US, 510300 CN CSI300)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPY_PATCH_DATE = pd.Timestamp("2015-04-09")


def daily_returns(close: pd.Series) -> pd.Series:
    out = close.astype(np.float64).pct_change()
    return out.replace([np.inf, -np.inf], np.nan)


def patch_spy_missing_day(df: pd.DataFrame) -> pd.DataFrame:
    """Insert 2015-04-09 with OHLC = median of adjacent sessions if absent."""
    out = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(out["date"])
    if (dates == SPY_PATCH_DATE).any():
        return out
    pos = int(dates.searchsorted(SPY_PATCH_DATE))
    if pos == 0 or pos >= len(out):
        raise ValueError(f"cannot patch SPY on {SPY_PATCH_DATE.date()}: out of range")
    prev = out.iloc[pos - 1]
    nxt = out.iloc[pos]
    patch = {col: float(np.median([prev[col], nxt[col]])) for col in ("open", "high", "low", "close")}
    row = {"date": SPY_PATCH_DATE, **patch}
    if "volume" in out.columns:
        row["volume"] = float(np.median([prev["volume"], nxt["volume"]]))
    return (
        pd.concat([out.iloc[:pos], pd.DataFrame([row]), out.iloc[pos:]], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_spy_daily(start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_us_daily(symbol="SPY", adjust="")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if df.empty:
        raise ValueError(f"empty SPY range {start}..{end}")
    df = patch_spy_missing_day(df)
    return df.sort_values("date").reset_index(drop=True)


def fetch_510300_daily(start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.fund_etf_hist_sina(symbol="sh510300")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if df.empty:
        raise ValueError(f"empty 510300 range {start}..{end}")
    return df.sort_values("date").reset_index(drop=True)


def forward_holding_returns(
    daily_ret: pd.Series,
    formation_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.Series:
    """Compound daily_ret over the next `horizon` sessions after each formation date."""
    cal = pd.DatetimeIndex(daily_ret.index.sort_values())
    daily_ret = daily_ret.reindex(cal)
    out: dict[pd.Timestamp, float] = {}
    for fd in pd.to_datetime(formation_dates):
        pos = cal.get_indexer([fd], method=None)[0]
        if pos < 0:
            raise ValueError(f"formation date {fd.date()} missing from ETF calendar")
        window = daily_ret.iloc[pos + 1 : pos + 1 + horizon]
        if len(window) < horizon:
            continue
        if window.isna().any():
            raise ValueError(f"NaN ETF daily return after {fd.date()}")
        out[pd.Timestamp(fd)] = float((1.0 + window).prod() - 1.0)
    if not out:
        raise ValueError("no ETF holding-period returns computed")
    return pd.Series(out).sort_index()


def period_volatility(returns: pd.Series) -> float:
    r = returns.astype(np.float64).dropna()
    if len(r) < 2:
        raise ValueError("need >= 2 returns for volatility")
    return float(r.std(ddof=1))


def scale_to_benchmark_vol(strategy_returns: pd.Series, benchmark_vol: float) -> pd.Series:
    strat_vol = period_volatility(strategy_returns)
    factor = benchmark_vol / strat_vol
    return strategy_returns * factor
