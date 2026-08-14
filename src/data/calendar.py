"""Market calendar and as_of sampling dates aligned with paper rebalancing."""

from __future__ import annotations

import pandas as pd

from config import SAMPLE_END, SAMPLE_START


def market_calendar(dates: pd.Series) -> pd.DatetimeIndex:
    unique = pd.DatetimeIndex(dates.unique()).sort_values()
    return unique


def _sample_period_end_dates(calendar: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading day in each pandas period, clipped to SAMPLE_START–SAMPLE_END.

    freq 'W' is week ending Sunday (Mon–Sun). The last session is usually Friday;
    a Thursday when Friday is a market holiday.
    """
    sample_start = pd.Timestamp(SAMPLE_START)
    sample_end = pd.Timestamp(SAMPLE_END)
    s = pd.Series(calendar, index=calendar)
    ends = s.groupby(s.dt.to_period(freq)).max()
    idx = pd.DatetimeIndex(ends.values).sort_values()
    return idx[(idx >= sample_start) & (idx <= sample_end)]


def as_of_dates_5d(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return _sample_period_end_dates(calendar, "W")


def as_of_dates_month_end(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return _sample_period_end_dates(calendar, "M")


def as_of_dates_quarter_end(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return _sample_period_end_dates(calendar, "Q")


def as_of_dates_for_window(window_days: int, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if window_days == 5:
        return as_of_dates_5d(calendar)
    if window_days == 20:
        return as_of_dates_month_end(calendar)
    if window_days == 60:
        return as_of_dates_quarter_end(calendar)
    raise ValueError(f"unsupported window_days={window_days}")


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
