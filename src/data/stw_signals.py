"""Signal engine for the 7,846 STW technical trading rules.

The engine is optimized for the replication use case:
compute all rule signals for one stock once, sample only the requested
rebalance dates, then let the pipeline write rule-column chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.stw_rules import STWRule

MISSING_SIGNAL = np.int8(127)


def _finite_price(close: np.ndarray) -> np.ndarray:
    return np.isfinite(close) & (close > 0.0)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rolling(window, min_periods=window)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def _rolling_max_prev(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rolling(window, min_periods=window)
        .max()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )


def _rolling_min_prev(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, copy=False)
        .rolling(window, min_periods=window)
        .min()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )


def _state_from_raw(raw: np.ndarray, hold: int | None = None) -> np.ndarray:
    """Convert entry signals {-1,0,1} into held positions.

    With no fixed holding period, the latest nonzero signal is carried
    forward.  With a fixed holding period, a new entry is held for exactly
    `hold` observations and intermediate signals are ignored.
    """
    out = np.zeros(len(raw), dtype=np.int8)
    pos = np.int8(0)
    lock = 0
    for i, sig in enumerate(raw.astype(np.int8, copy=False)):
        if hold is not None and lock > 0:
            out[i] = pos
            lock -= 1
            continue
        if sig != 0:
            pos = sig
            if hold is not None:
                lock = max(int(hold) - 1, 0)
        out[i] = pos
    return out


def _delay_raw(raw: np.ndarray, delay: int | None) -> np.ndarray:
    if delay is None:
        return raw.astype(np.int8, copy=False)
    out = np.zeros(len(raw), dtype=np.int8)
    long_run = 0
    short_run = 0
    d = int(delay)
    for i, sig in enumerate(raw.astype(np.int8, copy=False)):
        if sig > 0:
            long_run += 1
            short_run = 0
        elif sig < 0:
            short_run += 1
            long_run = 0
        else:
            long_run = 0
            short_run = 0
        if long_run >= d:
            out[i] = 1
        elif short_run >= d:
            out[i] = -1
    return out


def _raw_from_threshold(left: np.ndarray, right: np.ndarray, band: float = 0.0) -> np.ndarray:
    ok = np.isfinite(left) & np.isfinite(right)
    raw = np.zeros(len(left), dtype=np.int8)
    raw[ok & (left > right * (1.0 + band))] = 1
    raw[ok & (left < right * (1.0 - band))] = -1
    return raw


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    out = np.zeros(len(close), dtype=np.float64)
    if len(close) <= 1:
        return out
    delta = np.diff(close)
    vol = np.nan_to_num(volume[1:], nan=0.0, posinf=0.0, neginf=0.0)
    step = np.where(delta >= 0.0, vol, -vol)
    step[~np.isfinite(delta)] = 0.0
    out[1:] = np.cumsum(step)
    return out


@dataclass
class StockSignalContext:
    dates: pd.DatetimeIndex
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        self._ma_cache: dict[tuple[str, int], np.ndarray] = {}
        self._max_cache: dict[int, np.ndarray] = {}
        self._min_cache: dict[int, np.ndarray] = {}
        self._obv: np.ndarray | None = None

    @classmethod
    def from_stock_frame(cls, stock_df: pd.DataFrame) -> "StockSignalContext":
        df = stock_df.sort_values("DlyCalDt").drop_duplicates("DlyCalDt", keep="last")
        close = np.abs(df["DlyClose"].to_numpy(dtype=np.float64))
        volume = df["DlyVol"].to_numpy(dtype=np.float64)
        return cls(pd.DatetimeIndex(df["DlyCalDt"]), close, volume)

    @property
    def obv(self) -> np.ndarray:
        if self._obv is None:
            self._obv = _obv(self.close, self.volume)
        return self._obv

    def ma(self, source: str, window: int) -> np.ndarray:
        key = (source, int(window))
        if key not in self._ma_cache:
            base = self.close if source == "close" else self.obv
            self._ma_cache[key] = _rolling_mean(base, int(window))
        return self._ma_cache[key]

    def max_prev(self, window: int) -> np.ndarray:
        window = int(window)
        if window not in self._max_cache:
            self._max_cache[window] = _rolling_max_prev(self.close, window)
        return self._max_cache[window]

    def min_prev(self, window: int) -> np.ndarray:
        window = int(window)
        if window not in self._min_cache:
            self._min_cache[window] = _rolling_min_prev(self.close, window)
        return self._min_cache[window]

    def sampled_positions(self, daily_pos: np.ndarray, as_ofs: pd.DatetimeIndex) -> np.ndarray:
        locs = self.dates.get_indexer(as_ofs)
        out = np.full(len(as_ofs), MISSING_SIGNAL, dtype=np.int8)
        ok = locs >= 0
        out[ok] = daily_pos[locs[ok]].astype(np.int8, copy=False)
        return out


def _filter_state(ctx: StockSignalContext, rule: STWRule) -> np.ndarray:
    p = ctx.close
    ok = _finite_price(p)
    x = float(rule.params["x"])
    variant = rule.variant

    if variant == "extrema":
        e = int(rule.params["e"])
        lo = ctx.min_prev(e)
        hi = ctx.max_prev(e)
        raw = np.zeros(len(p), dtype=np.int8)
        valid = ok & np.isfinite(lo) & np.isfinite(hi)
        raw[valid & (p >= lo * (1.0 + x))] = 1
        raw[valid & (p <= hi * (1.0 - x))] = -1
        return _state_from_raw(raw)

    hold = int(rule.params["c"]) if variant == "hold" else None
    neutral_y = float(rule.params["y"]) if variant == "neutral" else None
    out = np.zeros(len(p), dtype=np.int8)
    if len(p) == 0:
        return out

    first = np.flatnonzero(ok)
    if len(first) == 0:
        return out
    ref_low = float(p[first[0]])
    ref_high = float(p[first[0]])
    pos = np.int8(0)
    lock = 0

    for i, price in enumerate(p):
        if not ok[i]:
            out[i] = pos
            continue

        if lock > 0:
            if pos > 0:
                ref_high = max(ref_high, float(price))
            elif pos < 0:
                ref_low = min(ref_low, float(price))
            out[i] = pos
            lock -= 1
            continue

        if pos == 0:
            ref_low = min(ref_low, float(price))
            ref_high = max(ref_high, float(price))
            if price >= ref_low * (1.0 + x):
                pos = np.int8(1)
                ref_high = float(price)
                lock = max(hold - 1, 0) if hold is not None else 0
            elif price <= ref_high * (1.0 - x):
                pos = np.int8(-1)
                ref_low = float(price)
                lock = max(hold - 1, 0) if hold is not None else 0
        elif pos > 0:
            ref_high = max(ref_high, float(price))
            unwind = neutral_y if neutral_y is not None else x
            if price <= ref_high * (1.0 - unwind):
                if neutral_y is None:
                    pos = np.int8(-1)
                    ref_low = float(price)
                    lock = max(hold - 1, 0) if hold is not None else 0
                else:
                    pos = np.int8(0)
                    ref_low = float(price)
                    ref_high = float(price)
        else:
            ref_low = min(ref_low, float(price))
            unwind = neutral_y if neutral_y is not None else x
            if price >= ref_low * (1.0 + unwind):
                if neutral_y is None:
                    pos = np.int8(1)
                    ref_high = float(price)
                    lock = max(hold - 1, 0) if hold is not None else 0
                else:
                    pos = np.int8(0)
                    ref_low = float(price)
                    ref_high = float(price)
        out[i] = pos
    return out


def _ma_like_state(ctx: StockSignalContext, rule: STWRule) -> np.ndarray:
    source = "obv" if rule.family == "OBV" else "close"
    base = ctx.obv if source == "obv" else ctx.close
    params = rule.params
    variant = rule.variant
    band = float(params.get("b", 0.0))
    delay = int(params["d"]) if "d" in params else None
    hold = int(params["c"]) if "c" in params else None

    if variant.startswith("cross"):
        fast = ctx.ma(source, int(params["short"]))
        slow = ctx.ma(source, int(params["long"]))
        raw = _raw_from_threshold(fast, slow, band)
    else:
        ma = ctx.ma(source, int(params["n"]))
        raw = _raw_from_threshold(base, ma, band)

    raw = _delay_raw(raw, delay)
    return _state_from_raw(raw, hold=hold)


def _sr_state(ctx: StockSignalContext, rule: STWRule) -> np.ndarray:
    p = ctx.close
    params = rule.params
    window = int(params.get("n", params.get("e")))
    hi = ctx.max_prev(window)
    lo = ctx.min_prev(window)
    band = float(params.get("b", 0.0))
    delay = int(params["d"]) if "d" in params else None
    hold = int(params["c"]) if "c" in params else None
    raw = np.zeros(len(p), dtype=np.int8)
    valid = _finite_price(p) & np.isfinite(hi) & np.isfinite(lo)
    raw[valid & (p > hi * (1.0 + band))] = 1
    raw[valid & (p < lo * (1.0 - band))] = -1
    raw = _delay_raw(raw, delay)
    pos = _state_from_raw(raw, hold=hold)

    if rule.variant == "range_stop":
        stop = float(params["stop"])
        stopped = pos.copy()
        entry_price = np.nan
        cur = np.int8(0)
        for i, side in enumerate(pos):
            price = p[i]
            if not np.isfinite(price):
                stopped[i] = cur
                continue
            if side != cur and side != 0:
                entry_price = price
                cur = side
            if cur > 0 and price <= entry_price * (1.0 - stop):
                cur = np.int8(0)
            elif cur < 0 and price >= entry_price * (1.0 + stop):
                cur = np.int8(0)
            stopped[i] = cur
        return stopped
    return pos


def _cb_state(ctx: StockSignalContext, rule: STWRule) -> np.ndarray:
    p = ctx.close
    params = rule.params
    n = int(params["n"])
    channel_x = float(params["x"])
    band = float(params.get("b", 0.0))
    hold = int(params["c"]) if "c" in params else None
    hi = ctx.max_prev(n)
    lo = ctx.min_prev(n)
    raw = np.zeros(len(p), dtype=np.int8)
    valid = _finite_price(p) & np.isfinite(hi) & np.isfinite(lo) & (lo > 0.0)
    channel = valid & (hi <= lo * (1.0 + channel_x))
    raw[channel & (p > hi * (1.0 + band))] = 1
    raw[channel & (p < lo * (1.0 - band))] = -1
    return _state_from_raw(raw, hold=hold)


def compute_rule_position(ctx: StockSignalContext, rule: STWRule) -> np.ndarray:
    if rule.family == "FR":
        return _filter_state(ctx, rule)
    if rule.family in ("MA", "OBV"):
        return _ma_like_state(ctx, rule)
    if rule.family == "SR":
        return _sr_state(ctx, rule)
    if rule.family == "CB":
        return _cb_state(ctx, rule)
    raise ValueError(f"unsupported STW family={rule.family}")


def compute_stock_rule_signals(
    stock_df: pd.DataFrame,
    rules: list[STWRule],
    as_ofs: pd.DatetimeIndex,
) -> pd.DataFrame:
    ctx = StockSignalContext.from_stock_frame(stock_df)
    as_ofs = pd.DatetimeIndex(as_ofs)
    data: dict[str, np.ndarray] = {"Date": as_ofs.to_numpy(dtype="datetime64[ns]")}
    for rule in rules:
        pos = compute_rule_position(ctx, rule)
        data[rule.name] = ctx.sampled_positions(pos, as_ofs)
    return pd.DataFrame(data)
