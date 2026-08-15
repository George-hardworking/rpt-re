"""Newey-West lag selection (fixed or automatic from sample size T)."""

from __future__ import annotations

from typing import Any

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
