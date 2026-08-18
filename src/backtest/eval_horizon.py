"""Evaluation horizons H1/H2/H3: signal at t, return at t+k rebalance periods."""

from __future__ import annotations

import pandas as pd

from backtest.io import attach_holding_return
from backtest.markets import MarketSpec


def align_panel_eval_horizon(
    panel: pd.DataFrame,
    spec: MarketSpec,
    eval_horizon: int,
    *,
    forward_return_on_formation_row: bool = True,
) -> pd.DataFrame:
    """Align formation-date signals with the eval_horizon-th future rebalance return.

    forward_return_on_formation_row=True (CNN / benchmark labels):
        Ret_Rd on row date t is the forward return from t+1; H1 needs no shift,
        H2 uses lag=1, H3 uses lag=2 on the rebalance calendar.

    forward_return_on_formation_row=False (CN factor returns table):
        Hk uses lag=k on the rebalance calendar (same as legacy --lag).
    """
    if eval_horizon < 1:
        raise ValueError(f"eval_horizon must be >= 1, got {eval_horizon}")
    ret_col = spec.ret_col
    if ret_col not in panel.columns:
        raise KeyError(f"panel missing {ret_col}")
    if eval_horizon == 1 and forward_return_on_formation_row:
        return panel

    lag = eval_horizon - 1 if forward_return_on_formation_row else eval_horizon
    if lag == 0:
        return panel

    rets = panel[[spec.id_col, spec.date_col, ret_col]].copy()
    signals = panel.drop(columns=[ret_col])
    return attach_holding_return(signals, rets, spec=spec, lag=lag)


def parse_eval_horizons(raw: list[int]) -> tuple[int, ...]:
    if not raw:
        raise ValueError("empty eval_horizons")
    out = tuple(raw)
    for h in out:
        if h < 1:
            raise ValueError(f"eval horizon must be >= 1, got {h}")
    return out


def backtest_output_stem(base: str, eval_horizon: int, *, direct_signal: bool = False) -> str:
    stem = f"direct_{base}" if direct_signal else base
    return f"{stem}_h{eval_horizon}"
