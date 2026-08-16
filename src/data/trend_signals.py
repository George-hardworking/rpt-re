"""Price-trend benchmark signals: MOM, reversals, 52-week high, HZZ combined trend.

All lookbacks use each stock's own trading-day sequence (log-sum rolling, no calendar fill).
Missing returns or prices in a window leave the signal NaN for that day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (
    CHAR_BETA_WINDOW,
    CHAR_LAG_WEEKLY_WINDOW,
    CHAR_LIQUIDITY_WINDOW,
    CHAR_VOL_WINDOW,
    LIQUIDITY_CHAR_COLS,
    TREND_52WH_WINDOW,
    TREND_HZZ_EMA_LAMBDA,
    TREND_HZZ_MA_LAGS,
    TREND_HZZ_MIN_NAMES,
    TREND_MOM_SKIP,
    TREND_MOM_WINDOW,
    TREND_STR_WINDOW,
    TREND_WSTR_WINDOW,
    TREND_SIGNALS_DAILY,
    market_processed_dir,
)

MA_COLS: tuple[str, ...] = tuple(f"ma{L}" for L in TREND_HZZ_MA_LAGS)
STOCK_SIGNAL_COLS: tuple[str, ...] = (
    "MOM",
    "REV1m_STR",
    "REV1w_WSTR",
    "DIST_52WH",
) + MA_COLS

DAILY_CHAR_COLS: tuple[str, ...] = STOCK_SIGNAL_COLS + LIQUIDITY_CHAR_COLS


def trend_signals_daily_root(market: str) -> Path:
    return market_processed_dir(market) / TREND_SIGNALS_DAILY


def trend_signal_partition_path(root: Path, permno: int) -> Path:
    return Path(root) / f"PERMNO={permno}" / "part-0.parquet"


def _parquet_column_names(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def trend_signal_partition_complete(root: Path, permno: int) -> bool:
    path = trend_signal_partition_path(root, permno)
    if not path.is_file():
        return False
    cols = _parquet_column_names(path)
    return all(c in cols for c in STOCK_SIGNAL_COLS)


def liquidity_partition_complete(root: Path, permno: int) -> bool:
    path = trend_signal_partition_path(root, permno)
    if not path.is_file():
        return False
    cols = _parquet_column_names(path)
    return all(c in cols for c in LIQUIDITY_CHAR_COLS)


def _rolling_log_cum(ret: pd.Series, window: int) -> np.ndarray:
    return ret.rolling(window, min_periods=window).sum().to_numpy(dtype=np.float64)


def stock_daily_trend_signals(stock_df: pd.DataFrame) -> pd.DataFrame:
    """One row per observed trading day with benchmark signal columns."""
    df = stock_df.sort_values("DlyCalDt")
    dates = pd.DatetimeIndex(df["DlyCalDt"])
    ret = df["DlyRet"].to_numpy(dtype=np.float64)
    close = np.abs(df["DlyClose"].to_numpy(dtype=np.float64))
    high = np.abs(df["DlyHigh"].to_numpy(dtype=np.float64))

    log1p = pd.Series(np.log1p(ret), index=dates)

    mom = np.expm1(_rolling_log_cum(log1p, TREND_MOM_WINDOW))
    mom = pd.Series(mom, index=dates).shift(TREND_MOM_SKIP).to_numpy(dtype=np.float64)

    rev1m = -np.expm1(_rolling_log_cum(log1p, TREND_STR_WINDOW))
    rev1w = -np.expm1(_rolling_log_cum(log1p, TREND_WSTR_WINDOW))

    close_s = pd.Series(close, index=dates)
    high_s = pd.Series(high, index=dates)
    roll_high = high_s.rolling(TREND_52WH_WINDOW, min_periods=TREND_52WH_WINDOW).max()
    dist_52wh = (close_s / roll_high).to_numpy(dtype=np.float64)

    out = pd.DataFrame({"DlyCalDt": dates})
    out["MOM"] = mom.astype(np.float32)
    out["REV1m_STR"] = rev1m.astype(np.float32)
    out["REV1w_WSTR"] = rev1w.astype(np.float32)
    out["DIST_52WH"] = dist_52wh.astype(np.float32)
    for lag in TREND_HZZ_MA_LAGS:
        ma = close_s.rolling(lag, min_periods=lag).mean()
        out[f"ma{lag}"] = (ma / close_s).to_numpy(dtype=np.float32)
    return out


def stock_liquidity_characteristics(
    stock_df: pd.DataFrame,
    mkt_ret: pd.Series,
) -> pd.DataFrame:
    """Beta, vol, dollar volume, zero-trade count, Amihud, lag weekly return."""
    df = stock_df.sort_values("DlyCalDt")
    dates = pd.DatetimeIndex(df["DlyCalDt"])
    ret = df["DlyRet"].to_numpy(dtype=np.float64)
    close = np.abs(df["DlyClose"].to_numpy(dtype=np.float64))
    vol = df["DlyVol"].to_numpy(dtype=np.float64)

    ret_s = pd.Series(ret, index=dates)
    mkt = mkt_ret.reindex(dates)
    roll_cov = ret_s.rolling(CHAR_BETA_WINDOW, min_periods=CHAR_BETA_WINDOW).cov(mkt)
    roll_var = mkt.rolling(CHAR_BETA_WINDOW, min_periods=CHAR_BETA_WINDOW).var()
    beta = (roll_cov / roll_var).to_numpy(dtype=np.float64)

    volatility = ret_s.rolling(CHAR_VOL_WINDOW, min_periods=CHAR_VOL_WINDOW).std().to_numpy(
        dtype=np.float64
    )

    dollar_vol = close * vol
    dollar_s = pd.Series(dollar_vol, index=dates)
    zero_s = pd.Series((vol <= 0.0) | ~np.isfinite(vol), index=dates).astype(float)
    zero_trade = zero_s.rolling(
        CHAR_LIQUIDITY_WINDOW, min_periods=CHAR_LIQUIDITY_WINDOW
    ).sum().to_numpy(dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        amihud_daily = np.abs(ret) / dollar_vol
    illiq = (
        pd.Series(amihud_daily, index=dates)
        .rolling(CHAR_LIQUIDITY_WINDOW, min_periods=CHAR_LIQUIDITY_WINDOW)
        .mean()
        .to_numpy(dtype=np.float64)
    )

    log1p = pd.Series(np.log1p(ret), index=dates)
    lag_weekly = np.expm1(
        log1p.rolling(CHAR_LAG_WEEKLY_WINDOW, min_periods=CHAR_LAG_WEEKLY_WINDOW).sum()
    ).to_numpy(dtype=np.float64)

    out = pd.DataFrame({"DlyCalDt": dates})
    out["Beta"] = beta.astype(np.float32)
    out["Volatility"] = volatility.astype(np.float32)
    out["DollarVol"] = dollar_s.to_numpy(dtype=np.float32)
    out["ZeroTrade"] = zero_trade.astype(np.float32)
    out["Illiquidity"] = illiq.astype(np.float32)
    out["LagWeeklyRet"] = lag_weekly.astype(np.float32)
    return out


def append_liquidity_characteristics(
    stock_df: pd.DataFrame,
    permno: int,
    output_root: Path,
    mkt_ret: pd.Series,
) -> Path:
    path = trend_signal_partition_path(output_root, permno)
    if path.is_file():
        panel = pd.read_parquet(path)
        panel["DlyCalDt"] = pd.to_datetime(panel["DlyCalDt"])
    else:
        panel = stock_daily_trend_signals(stock_df)
        panel.insert(0, "PERMNO", np.int64(permno))

    liq = stock_liquidity_characteristics(stock_df, mkt_ret)
    for col in LIQUIDITY_CHAR_COLS:
        panel[col] = liq[col].to_numpy(dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path


def write_stock_trend_signals(
    stock_df: pd.DataFrame,
    permno: int,
    output_root: Path,
) -> Path:
    panel = stock_daily_trend_signals(stock_df)
    panel.insert(0, "PERMNO", np.int64(permno))
    path = trend_signal_partition_path(output_root, permno)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path


def read_stock_trend_signals(signals_root: Path, permno: int) -> pd.DataFrame:
    path = trend_signal_partition_path(signals_root, permno)
    if not path.is_file():
        raise FileNotFoundError(f"missing trend signals: {path}")
    df = pd.read_parquet(path)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    return df.sort_values("DlyCalDt")


def join_signals_at_dates(
    signals: pd.DataFrame,
    as_ofs: pd.DatetimeIndex,
    *,
    cols: tuple[str, ...],
) -> pd.DataFrame:
    """Point-in-time join: signal values at each as_of date (no forward fill)."""
    indexed = signals.set_index("DlyCalDt")
    idx = pd.DatetimeIndex(as_ofs)
    subset = indexed.reindex(idx)[list(cols)]
    out = subset.reset_index(names="Date")
    return out


def compute_hzz_trend_scores(
    panel: pd.DataFrame,
    *,
    date_col: str = "Date",
    ret_col: str,
    ma_cols: tuple[str, ...] = MA_COLS,
    lambda_ema: float = TREND_HZZ_EMA_LAMBDA,
    min_names: int = TREND_HZZ_MIN_NAMES,
) -> pd.Series:
    """Cross-sectional HZZ trend score; regression uses prior rebalance return and MA ratios."""
    dates = pd.DatetimeIndex(sorted(panel[date_col].unique()))
    beta_ema: np.ndarray | None = None
    score_parts: list[pd.Series] = []

    for i, d in enumerate(dates):
        cur_mask = panel[date_col] == d
        cur_idx = panel.index[cur_mask]
        if i == 0:
            score_parts.append(pd.Series(np.nan, index=cur_idx, dtype=np.float64))
            continue

        d_prev = dates[i - 1]
        prev = panel.loc[panel[date_col] == d_prev, [ret_col, *ma_cols]].dropna()
        if len(prev) < min_names:
            score_parts.append(pd.Series(np.nan, index=cur_idx, dtype=np.float64))
            continue

        y = prev[ret_col].to_numpy(dtype=np.float64)
        x = prev[list(ma_cols)].to_numpy(dtype=np.float64)
        design = np.column_stack([np.ones(len(y), dtype=np.float64), x])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        if beta_ema is None:
            beta_ema = beta.copy()
        else:
            beta_ema = (1.0 - lambda_ema) * beta_ema + lambda_ema * beta

        cur = panel.loc[cur_mask, list(ma_cols)]
        x_cur = cur.to_numpy(dtype=np.float64)
        design_cur = np.column_stack([np.ones(len(x_cur), dtype=np.float64), x_cur])
        scores = design_cur @ beta_ema
        score_parts.append(pd.Series(scores, index=cur_idx, dtype=np.float64))

    return pd.concat(score_parts).sort_index()
