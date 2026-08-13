"""Render OHLC + MA + volume images from point-in-time window data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import COLS_PER_DAY, IMAGE_LAYOUT, PIXEL_OFF, PIXEL_ON
from data.calendar import window_calendar_days


@dataclass(frozen=True)
class WindowArrays:
    dates: pd.DatetimeIndex
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ret: np.ndarray
    volume: np.ndarray


def _finite(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def build_window_arrays(
    stock_df: pd.DataFrame,
    window_dates: pd.DatetimeIndex,
) -> WindowArrays:
    indexed = stock_df.set_index("DlyCalDt")
    rows = indexed.reindex(window_dates)
    return WindowArrays(
        dates=window_dates,
        open_=rows["DlyOpen"].to_numpy(dtype=np.float64),
        high=rows["DlyHigh"].to_numpy(dtype=np.float64),
        low=rows["DlyLow"].to_numpy(dtype=np.float64),
        close=rows["DlyClose"].to_numpy(dtype=np.float64),
        ret=rows["DlyRet"].to_numpy(dtype=np.float64),
        volume=rows["DlyVol"].to_numpy(dtype=np.float64),
    )


def history_arrays(
    stock_df: pd.DataFrame,
    history_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    indexed = stock_df.set_index("DlyCalDt")
    rows = indexed.reindex(history_dates)
    return (
        rows["DlyClose"].to_numpy(dtype=np.float64),
        rows["DlyRet"].to_numpy(dtype=np.float64),
    )


def adjusted_close_window_scale(
    history_close: np.ndarray,
    history_ret: np.ndarray,
    window_close: np.ndarray,
    window_ret: np.ndarray,
) -> np.ndarray:
    combined_close = np.concatenate([history_close, window_close])
    combined_ret = np.concatenate([history_ret, window_ret])
    n_hist = len(history_close)
    n = len(window_close)
    total = n_hist + n
    adj = np.full(total, np.nan, dtype=np.float64)

    if not _finite(window_close[0]):
        return adj[-n:]

    adj[n_hist] = 1.0
    for i in range(n_hist + 1, total):
        if _finite(combined_ret[i]) and _finite(adj[i - 1]):
            adj[i] = adj[i - 1] * (1.0 + combined_ret[i])

    for i in range(n_hist - 1, -1, -1):
        if _finite(combined_ret[i + 1]) and _finite(adj[i + 1]):
            adj[i] = adj[i + 1] / (1.0 + combined_ret[i + 1])

    return adj[-n:]


def moving_average_window_scale(
    adj_full: np.ndarray,
    n_hist: int,
    window_days: int,
) -> np.ndarray:
    n = len(adj_full) - n_hist
    ma = np.full(n, np.nan, dtype=np.float64)
    for k in range(n):
        i = n_hist + k
        start = i - window_days + 1
        if start < 0:
            continue
        segment = adj_full[start : i + 1]
        if np.all(_finite(segment)):
            ma[k] = segment.mean()
    return ma


def adjusted_ohlc_for_window(
    window: WindowArrays,
    history_close: np.ndarray,
    history_ret: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(window.dates)
    window_days = n
    adj_open = np.full(n, np.nan, dtype=np.float64)
    adj_high = np.full(n, np.nan, dtype=np.float64)
    adj_low = np.full(n, np.nan, dtype=np.float64)
    adj_close = np.full(n, np.nan, dtype=np.float64)

    if not _finite(window.close[0]):
        return adj_open, adj_high, adj_low, adj_close, np.full(n, np.nan)

    adj_close = adjusted_close_window_scale(
        history_close, history_ret, window.close, window.ret
    )

    for i in range(n):
        c_raw = window.close[i]
        c_adj = adj_close[i]
        if not _finite(c_raw) or not _finite(c_adj) or c_raw == 0.0:
            continue
        if _finite(window.open_[i]):
            adj_open[i] = c_adj * (window.open_[i] / c_raw)
        if _finite(window.high[i]):
            adj_high[i] = c_adj * (window.high[i] / c_raw)
        if _finite(window.low[i]):
            adj_low[i] = c_adj * (window.low[i] / c_raw)

    combined_close = np.concatenate([history_close, window.close])
    combined_ret = np.concatenate([history_ret, window.ret])
    n_hist = len(history_close)
    adj_full = np.full(len(combined_close), np.nan, dtype=np.float64)
    if _finite(window.close[0]):
        adj_full[n_hist] = 1.0
        for i in range(n_hist + 1, len(combined_close)):
            if _finite(combined_ret[i]) and _finite(adj_full[i - 1]):
                adj_full[i] = adj_full[i - 1] * (1.0 + combined_ret[i])
        for i in range(n_hist - 1, -1, -1):
            if _finite(combined_ret[i + 1]) and _finite(adj_full[i + 1]):
                adj_full[i] = adj_full[i + 1] / (1.0 + combined_ret[i + 1])

    ma = moving_average_window_scale(adj_full, n_hist, window_days)
    return adj_open, adj_high, adj_low, adj_close, ma


def stock_first_date(stock_df: pd.DataFrame) -> pd.Timestamp:
    return stock_df["DlyCalDt"].min()


def stock_last_date(stock_df: pd.DataFrame) -> pd.Timestamp:
    return stock_df["DlyCalDt"].max()


def is_ipo_in_window(first_date: pd.Timestamp, window_start: pd.Timestamp) -> bool:
    return first_date > window_start


def is_delist_in_window(
    last_date: pd.Timestamp,
    window_start: pd.Timestamp,
    as_of: pd.Timestamp,
    calendar_last: pd.Timestamp,
) -> bool:
    if last_date >= calendar_last:
        return False
    return window_start <= last_date <= as_of


def _price_to_rows(
    values: np.ndarray,
    pmin: float,
    prange: float,
    price_rows: int,
    *,
    clip: bool = True,
) -> np.ndarray:
    norm = (values - pmin) / prange
    if clip:
        norm = np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0)
    return np.round((1.0 - norm) * (price_rows - 1)).astype(np.intp)


def _draw_line(
    image: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    row_hi: int,
) -> None:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    _, width = image.shape
    while True:
        if 0 <= r < row_hi and 0 <= c < width:
            image[r, c] = PIXEL_ON
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def render_image(
    adj_open: np.ndarray,
    adj_high: np.ndarray,
    adj_low: np.ndarray,
    adj_close: np.ndarray,
    ma: np.ndarray,
    volume: np.ndarray,
    window_days: int,
) -> Optional[np.ndarray]:
    price_rows, volume_rows = IMAGE_LAYOUT[window_days]
    height = price_rows + volume_rows
    width = window_days * COLS_PER_DAY
    image = np.zeros((height, width), dtype=np.uint8)

    fin = _finite
    price_values = []
    for arr in (adj_open, adj_high, adj_low, adj_close):
        m = fin(arr)
        if m.any():
            price_values.append(arr[m])
    if not price_values:
        return None

    stacked = np.concatenate(price_values)
    pmin = stacked.min()
    pmax = stacked.max()
    prange = pmax - pmin
    if prange == 0.0:
        prange = 1e-6

    vol_panel_start = price_rows
    vol_valid = volume[fin(volume) & (volume > 0)]
    vmax = vol_valid.max() if vol_valid.size else 0.0

    days = np.arange(window_days)
    col_open = days * COLS_PER_DAY
    col_bar = col_open + 1
    col_close = col_open + 2
    rows = np.arange(height)[:, None]

    has_high = fin(adj_high)
    has_low = fin(adj_low)
    has_open = fin(adj_open)
    has_close = fin(adj_close)
    bar_valid = has_high & has_low

    high_rows = _price_to_rows(adj_high, pmin, prange, price_rows)
    low_rows = _price_to_rows(adj_low, pmin, prange, price_rows)
    top = np.minimum(high_rows, low_rows)
    bottom = np.maximum(high_rows, low_rows)
    bar_mask = bar_valid & (rows >= top) & (rows <= bottom)
    rr, dd = np.nonzero(bar_mask)
    image[rr, col_bar[dd]] = PIXEL_ON

    open_draw = bar_valid & has_open
    od = np.nonzero(open_draw)[0]
    if od.size:
        image[_price_to_rows(adj_open[od], pmin, prange, price_rows), col_open[od]] = PIXEL_ON

    close_draw = bar_valid & has_close
    cd = np.nonzero(close_draw)[0]
    if cd.size:
        image[_price_to_rows(adj_close[cd], pmin, prange, price_rows), col_close[cd]] = PIXEL_ON

    vol_draw = (vmax > 0.0) & fin(volume) & (volume > 0.0)
    if vol_draw.any():
        vol_norm = np.clip(volume / vmax, 0.0, 1.0)
        vol_height = np.maximum(1, np.round(vol_norm * (volume_rows - 1)).astype(np.intp))
        start_row = height - 1
        end_row = np.maximum(vol_panel_start, start_row - vol_height + 1)
        vol_mask = vol_draw & (rows >= end_row) & (rows <= start_row)
        vr, vd = np.nonzero(vol_mask)
        image[vr, col_bar[vd]] = PIXEL_ON

    ma_finite_mask = fin(ma)
    if ma_finite_mask.any():
        ma_days = np.nonzero(ma_finite_mask)[0]
        ma_rows = _price_to_rows(ma[ma_days], pmin, prange, price_rows, clip=False)
        ma_cols = ma_days * COLS_PER_DAY + 1
        for i in range(1, len(ma_days)):
            _draw_line(
                image,
                int(ma_rows[i - 1]),
                int(ma_cols[i - 1]),
                int(ma_rows[i]),
                int(ma_cols[i]),
                row_hi=price_rows,
            )

    return image


def image_shape(window_days: int) -> tuple[int, int]:
    price_rows, volume_rows = IMAGE_LAYOUT[window_days]
    height = price_rows + volume_rows
    width = window_days * COLS_PER_DAY
    return height, width


def try_build_window(
    stock_df: pd.DataFrame,
    permno: int,
    as_of: pd.Timestamp,
    window_days: int,
    calendar: pd.DatetimeIndex,
    calendar_last: pd.Timestamp,
) -> Optional[tuple[np.ndarray, dict]]:
    window_dates = window_calendar_days(as_of, window_days, calendar)
    window_start = window_dates[0]
    first_date = stock_first_date(stock_df)
    last_date = stock_last_date(stock_df)

    if is_ipo_in_window(first_date, window_start):
        return None
    if is_delist_in_window(last_date, window_start, as_of, calendar_last):
        return None

    start_pos = calendar.searchsorted(window_start)
    hist_start_pos = start_pos - (window_days - 1)
    if hist_start_pos < 0:
        return None
    history_dates = calendar[hist_start_pos : start_pos]

    window = build_window_arrays(stock_df, window_dates)
    history_close, history_ret = history_arrays(stock_df, history_dates)

    adj_open, adj_high, adj_low, adj_close, ma = adjusted_ohlc_for_window(
        window, history_close, history_ret
    )
    image = render_image(
        adj_open, adj_high, adj_low, adj_close, ma, window.volume, window_days
    )
    if image is None:
        return None

    return image, {"Date": as_of, "StockID": permno}
