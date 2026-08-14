"""Image-side labels matching author GenerateStockData._generate_daily_features.

Forward returns use the stock's own subsequent trading days (no fill). EWMA vol is
point-in-time through as_of. Classification: >0 → 1, <=0 → 0, missing → 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import EWMA_VOL_SPAN, LABEL_HORIZONS

PERIOD_HORIZON = {
    "Ret_week": 5,
    "Ret_month": 20,
    "Ret_quarter": 60,
    "Ret_year": 250,
}


def _ret_label(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), 2, dtype=np.int8)
    finite = np.isfinite(values)
    out[finite & (values > 0)] = 1
    out[finite & (values <= 0)] = 0
    return out


def _tstat(values: np.ndarray, vol: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    ok = np.isfinite(vol) & (vol != 0.0)
    out[ok] = (values[ok] / vol[ok]).astype(np.float32)
    return out


def stock_label_panel(stock_df: pd.DataFrame) -> pd.DataFrame:
    """One row per observed date: MarketCap, EWMA_vol, Ret / Ret_Nd, labels, t-stats."""
    df = stock_df.sort_values("DlyCalDt")
    dates = pd.DatetimeIndex(df["DlyCalDt"])
    ret = df["DlyRet"].to_numpy(dtype=np.float64)

    log1p = np.log1p(ret)
    log_s = pd.Series(log1p)
    fwd: dict[int, np.ndarray] = {}
    for h in LABEL_HORIZONS:
        fwd_log = log_s.rolling(window=h, min_periods=h).sum().shift(-h)
        fwd[h] = np.expm1(fwd_log.to_numpy(dtype=np.float64))

    vol = (
        pd.Series(ret)
        .ewm(span=EWMA_VOL_SPAN, min_periods=EWMA_VOL_SPAN)
        .std()
        .to_numpy(dtype=np.float64)
    )

    panel = pd.DataFrame(index=dates)
    panel["EWMA_vol"] = vol.astype(np.float32)
    panel["Ret"] = ret.astype(np.float32)
    panel["MarketCap"] = df["DlyCap"].to_numpy(dtype=np.float32)
    panel["Ret_tstat"] = _tstat(ret, vol)

    for h in LABEL_HORIZONS:
        name = f"Ret_{h}d"
        panel[name] = fwd[h].astype(np.float32)
        panel[f"{name}_tstat"] = _tstat(fwd[h], vol)
        panel[f"{name}_label"] = _ret_label(fwd[h])

    for period_name, h in PERIOD_HORIZON.items():
        panel[period_name] = panel[f"Ret_{h}d"]

    panel["Ret_label"] = _ret_label(ret)
    return panel


def build_stock_features(stock_df: pd.DataFrame, permno: int) -> pd.DataFrame:
    """Daily feature panel for one stock; one row per observed date."""
    panel = stock_label_panel(stock_df)
    out = panel.reset_index(names="DlyCalDt")
    out.insert(0, "PERMNO", permno)
    return out


def labels_for_as_of(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    permno: int,
    window_days: int,
) -> dict:
    rec = panel.loc[as_of].to_dict()
    rec["Date"] = pd.Timestamp(as_of)
    rec["StockID"] = permno
    rec["window_size"] = np.uint8(window_days)
    return rec


def labels_for_as_ofs(
    panel: pd.DataFrame,
    as_ofs: list[pd.Timestamp] | pd.DatetimeIndex,
    permno: int,
    window_days: int,
) -> list[dict]:
    if len(as_ofs) == 0:
        return []

    subset = panel.loc[as_ofs]
    if isinstance(subset, pd.Series):
        subset = subset.to_frame().T

    out: list[dict] = []
    for as_of, (_, row) in zip(as_ofs, subset.iterrows()):
        rec = row.to_dict()
        rec["Date"] = pd.Timestamp(as_of)
        rec["StockID"] = permno
        rec["window_size"] = np.uint8(window_days)
        out.append(rec)
    return out
