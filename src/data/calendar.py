"""Market calendar and as_of sampling dates aligned with paper rebalancing."""

from __future__ import annotations

import pandas as pd

from config import (
    MARKET_US,
    SAMPLE_END,
    SAMPLE_START,
    WINDOW_DEFAULT_SAMPLE_FREQ,
    market_sample_config,
)


def market_calendar(dates: pd.Series) -> pd.DatetimeIndex:
    unique = pd.DatetimeIndex(dates.unique()).sort_values()
    return unique


def _sample_period_end_dates(
    calendar: pd.DatetimeIndex,
    freq: str,
    *,
    sample_start: pd.Timestamp | None = None,
    sample_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Last trading day in each pandas period, clipped to sample window.

    freq 'W' is week ending Sunday (Mon–Sun). The last session is usually Friday;
    a Thursday when Friday is a market holiday.
    """
    lo = pd.Timestamp(sample_start if sample_start is not None else SAMPLE_START)
    hi = pd.Timestamp(sample_end if sample_end is not None else SAMPLE_END)
    s = pd.Series(calendar, index=calendar)
    ends = s.groupby(s.dt.to_period(freq)).max()
    idx = pd.DatetimeIndex(ends.values).sort_values()
    return idx[(idx >= lo) & (idx <= hi)]


def as_of_dates_week(
    calendar: pd.DatetimeIndex,
    *,
    sample_start: pd.Timestamp | None = None,
    sample_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    return _sample_period_end_dates(
        calendar, "W", sample_start=sample_start, sample_end=sample_end
    )


def as_of_dates_month_end(
    calendar: pd.DatetimeIndex,
    *,
    sample_start: pd.Timestamp | None = None,
    sample_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    return _sample_period_end_dates(
        calendar, "M", sample_start=sample_start, sample_end=sample_end
    )


def as_of_dates_quarter_end(
    calendar: pd.DatetimeIndex,
    *,
    sample_start: pd.Timestamp | None = None,
    sample_end: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    return _sample_period_end_dates(
        calendar, "Q", sample_start=sample_start, sample_end=sample_end
    )


def as_of_dates_for_freq(
    sample_freq: str,
    calendar: pd.DatetimeIndex,
    *,
    market: str = MARKET_US,
) -> pd.DatetimeIndex:
    cfg = market_sample_config(market)
    sample_start = pd.Timestamp(cfg.sample_start)
    sample_end = pd.Timestamp(cfg.sample_end)
    if sample_freq == "week":
        return as_of_dates_week(calendar, sample_start=sample_start, sample_end=sample_end)
    if sample_freq == "month":
        return as_of_dates_month_end(
            calendar, sample_start=sample_start, sample_end=sample_end
        )
    if sample_freq == "quarter":
        return as_of_dates_quarter_end(
            calendar, sample_start=sample_start, sample_end=sample_end
        )
    raise ValueError(f"unsupported sample_freq={sample_freq}")


def as_of_dates_for_window(window_days: int, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Legacy: sample dates tied to image window (diagonal bundles only)."""
    return as_of_dates_for_freq(WINDOW_DEFAULT_SAMPLE_FREQ[window_days], calendar)


def window_calendar_days(
    as_of: pd.Timestamp,
    window_days: int,
    calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    pos = calendar.searchsorted(as_of, side="right") - 1
    if pos < window_days - 1:
        raise ValueError(f"insufficient calendar history before as_of={as_of}")
    if calendar[pos] != as_of:
        raise ValueError(f"as_of={as_of} is not a trading day on the market calendar")
    start_pos = pos - window_days + 1
    return calendar[start_pos : pos + 1]


def forward_calendar_days(
    as_of: pd.Timestamp,
    horizon: int,
    calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    pos = calendar.searchsorted(as_of, side="right") - 1
    if calendar[pos] != as_of:
        raise ValueError(f"as_of={as_of} is not a trading day on the market calendar")
    end_pos = pos + horizon
    if end_pos >= len(calendar):
        raise IndexError(f"horizon {horizon} exceeds calendar after as_of={as_of}")
    return calendar[pos + 1 : end_pos + 1]
