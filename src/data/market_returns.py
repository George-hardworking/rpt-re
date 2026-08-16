"""Value-weighted market daily returns from OHLC partitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import market_vw_returns_path


def _cap_col(columns: set[str]) -> str:
    if "DlyTotalCap" in columns:
        return "DlyTotalCap"
    return "DlyCap"


def compute_market_vw_returns(ohlc_path: Path) -> pd.DataFrame:
    """One row per trading day: cap-weighted average of DlyRet."""
    ohlc_path = Path(ohlc_path)
    files = sorted(ohlc_path.glob("PERMNO=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no OHLC partitions under {ohlc_path}")

    cap_col = _cap_col(set(pq.read_schema(files[0]).names))
    num: dict[pd.Timestamp, float] = {}
    den: dict[pd.Timestamp, float] = {}

    for path in files:
        df = pd.read_parquet(path, columns=["DlyCalDt", "DlyRet", cap_col])
        dates = pd.to_datetime(df["DlyCalDt"])
        ret = df["DlyRet"].to_numpy(dtype=np.float64)
        cap = df[cap_col].to_numpy(dtype=np.float64)
        for d, r, c in zip(dates, ret, cap):
            if not np.isfinite(r) or not np.isfinite(c) or c <= 0.0:
                continue
            ts = pd.Timestamp(d)
            num[ts] = num.get(ts, 0.0) + c * r
            den[ts] = den.get(ts, 0.0) + c

    if not num:
        raise ValueError(f"empty market return aggregation from {ohlc_path}")

    dates = sorted(num.keys())
    mkt_ret = np.array([num[d] / den[d] for d in dates], dtype=np.float32)
    return pd.DataFrame({"DlyCalDt": dates, "MktRet": mkt_ret})


def write_market_vw_returns(
    ohlc_path: Path,
    market: str,
    *,
    out_path: Path | None = None,
) -> Path:
    path = out_path if out_path is not None else market_vw_returns_path(market)
    path = Path(path)
    panel = compute_market_vw_returns(ohlc_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return path


def read_market_vw_returns(market: str, *, path: Path | None = None) -> pd.Series:
    p = path if path is not None else market_vw_returns_path(market)
    df = pd.read_parquet(p)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    return df.set_index("DlyCalDt")["MktRet"].sort_index()
