"""Render OHLC + MA + volume images from point-in-time window data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import COLS_PER_DAY, IMAGE_LAYOUT, PIXEL_OFF, PIXEL_ON


@dataclass(frozen=True)
class WindowArrays:
    dates: pd.DatetimeIndex
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ret: np.ndarray
    volume: np.ndarray


@dataclass(frozen=True)
class StockOHLC:
    dates: pd.DatetimeIndex
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    ret: np.ndarray
    volume: np.ndarray


def prepare_stock_ohlc(stock_df: pd.DataFrame) -> StockOHLC:
    dates = pd.DatetimeIndex(stock_df["DlyCalDt"])
    return StockOHLC(
        dates=dates,
        first_date=dates[0],
        last_date=dates[-1],
        open_=np.abs(stock_df["DlyOpen"].to_numpy(dtype=np.float64)),
        high=np.abs(stock_df["DlyHigh"].to_numpy(dtype=np.float64)),
        low=np.abs(stock_df["DlyLow"].to_numpy(dtype=np.float64)),
        close=np.abs(stock_df["DlyClose"].to_numpy(dtype=np.float64)),
        ret=stock_df["DlyRet"].to_numpy(dtype=np.float64),
        volume=stock_df["DlyVol"].to_numpy(dtype=np.float64),
    )


def _window_arrays_from_slice(ohlc: StockOHLC, sl: slice) -> WindowArrays:
    return WindowArrays(
        dates=ohlc.dates[sl],
        open_=ohlc.open_[sl],
        high=ohlc.high[sl],
        low=ohlc.low[sl],
        close=ohlc.close[sl],
        ret=ohlc.ret[sl],
        volume=ohlc.volume[sl],
    )


def _finite(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def build_window_arrays(
    stock_df: pd.DataFrame,
    window_dates: pd.DatetimeIndex,
) -> WindowArrays:
    indexed = stock_df.set_index("DlyCalDt")
    rows = indexed.reindex(window_dates)
    # CRSP: negative price flags a bid/ask midpoint; magnitude is the price.
    return WindowArrays(
        dates=window_dates,
        open_=np.abs(rows["DlyOpen"].to_numpy(dtype=np.float64)),
        high=np.abs(rows["DlyHigh"].to_numpy(dtype=np.float64)),
        low=np.abs(rows["DlyLow"].to_numpy(dtype=np.float64)),
        close=np.abs(rows["DlyClose"].to_numpy(dtype=np.float64)),
        ret=rows["DlyRet"].to_numpy(dtype=np.float64),
        volume=rows["DlyVol"].to_numpy(dtype=np.float64),
    )


CHART_CLOSE_DECIMALS = 3


def chain_adjust_from_first_day(series: WindowArrays) -> Optional[tuple[np.ndarray, ...]]:
    """Author adjust_price: missing Ret leaves that day blank and keeps the last valid close."""
    n = len(series.dates)
    adj_open = np.full(n, np.nan, dtype=np.float64)
    adj_high = np.full(n, np.nan, dtype=np.float64)
    adj_low = np.full(n, np.nan, dtype=np.float64)
    adj_close = np.full(n, np.nan, dtype=np.float64)

    first_close = series.close[0]
    if not _finite(first_close) or first_close == 0.0:
        return None

    adj_close[0] = 1.0
    adj_open[0] = series.open_[0] / first_close
    adj_high[0] = series.high[0] / first_close
    adj_low[0] = series.low[0] / first_close
    pre_close = 1.0

    for i in range(1, n):
        today_ret = series.ret[i]
        if np.isnan(today_ret):
            continue
        adj_close[i] = (1.0 + today_ret) * pre_close
        today_closep = series.close[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            adj_open[i] = adj_close[i] / today_closep * series.open_[i]
            adj_high[i] = adj_close[i] / today_closep * series.high[i]
            adj_low[i] = adj_close[i] / today_closep * series.low[i]
        if not np.isnan(adj_close[i]):
            pre_close = float(adj_close[i])

    return adj_open, adj_high, adj_low, adj_close


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
    series: WindowArrays,
    n_hist: int,
    draw_ma: bool,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Chain from slice day 0, then divide by chart-window first close so that day is 1."""
    chained = chain_adjust_from_first_day(series)
    if chained is None:
        return None
    adj_open, adj_high, adj_low, adj_close = chained

    scale = adj_close[n_hist]
    if not _finite(scale) or scale == 0.0:
        return None
    adj_open = adj_open / scale
    adj_high = adj_high / scale
    adj_low = adj_low / scale
    adj_close = adj_close / scale
    if np.around(adj_close[n_hist], decimals=CHART_CLOSE_DECIMALS) != 1.0:
        return None

    window_days = len(series.dates) - n_hist
    if draw_ma:
        ma = moving_average_window_scale(adj_close, n_hist, window_days)
    else:
        ma = np.full(window_days, np.nan, dtype=np.float64)

    return (
        adj_open[n_hist:],
        adj_high[n_hist:],
        adj_low[n_hist:],
        adj_close[n_hist:],
        ma,
    )


def observed_dates(stock_df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stock_df["DlyCalDt"]).sort_values()


def window_stock_days(
    dates: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    window_days: int,
) -> Optional[pd.DatetimeIndex]:
    loc = dates.get_indexer([as_of])[0]
    if loc < 0:
        return None
    start = loc - window_days + 1
    if start < 0:
        return None
    return dates[start : loc + 1]


def history_stock_days(
    dates: pd.DatetimeIndex,
    window_start: pd.Timestamp,
    n_hist: int,
) -> pd.DatetimeIndex:
    loc = dates.get_indexer([window_start])[0]
    start = loc - n_hist
    if start < 0:
        return dates[:0]
    return dates[start:loc]


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
    *,
    middle_col_only: bool = False,
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
            if not middle_col_only or c % COLS_PER_DAY == 1:
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
        ma_cols_mid = ma_days * COLS_PER_DAY + 1
        ma_cols_open = ma_days * COLS_PER_DAY
        ma_cols_close = ma_days * COLS_PER_DAY + 2
        image[ma_rows, ma_cols_open] = PIXEL_ON
        image[ma_rows, ma_cols_mid] = PIXEL_ON
        image[ma_rows, ma_cols_close] = PIXEL_ON
        for i in range(1, len(ma_days)):
            _draw_line(
                image,
                int(ma_rows[i - 1]),
                int(ma_cols_mid[i - 1]),
                int(ma_rows[i]),
                int(ma_cols_mid[i]),
                row_hi=price_rows,
                middle_col_only=True,
            )

    return image


def image_shape(window_days: int) -> tuple[int, int]:
    price_rows, volume_rows = IMAGE_LAYOUT[window_days]
    height = price_rows + volume_rows
    width = window_days * COLS_PER_DAY
    return height, width


def try_build_window_from_ohlc(
    ohlc: StockOHLC,
    permno: int,
    as_of: pd.Timestamp,
    window_days: int,
    calendar_last: pd.Timestamp,
    *,
    loc: int | None = None,
) -> Optional[tuple[np.ndarray, dict]]:
    if loc is None:
        loc = ohlc.dates.get_indexer([as_of])[0]
        if loc < 0:
            return None

    start = loc - window_days + 1
    if start < 0:
        return None

    window_start = ohlc.dates[start]
    if is_ipo_in_window(ohlc.first_date, window_start):
        return None
    if is_delist_in_window(ohlc.last_date, window_start, as_of, calendar_last):
        return None

    ma_offset = window_days
    hist_start = start - ma_offset
    draw_ma = hist_start >= 0
    if draw_ma:
        slice_start = hist_start
        n_hist = ma_offset
    else:
        slice_start = start
        n_hist = 0

    series = _window_arrays_from_slice(ohlc, slice(slice_start, loc + 1))
    built_adj = adjusted_ohlc_for_window(series, n_hist, draw_ma)
    if built_adj is None:
        return None
    adj_open, adj_high, adj_low, adj_close, ma = built_adj
    image = render_image(
        adj_open, adj_high, adj_low, adj_close, ma, series.volume[n_hist:], window_days
    )
    if image is None:
        return None

    return image, {"Date": as_of, "StockID": permno}


def try_build_window(
    stock_df: pd.DataFrame,
    permno: int,
    as_of: pd.Timestamp,
    window_days: int,
    calendar_last: pd.Timestamp,
) -> Optional[tuple[np.ndarray, dict]]:
    ohlc = prepare_stock_ohlc(stock_df)
    return try_build_window_from_ohlc(
        ohlc, permno, as_of, window_days, calendar_last
    )
