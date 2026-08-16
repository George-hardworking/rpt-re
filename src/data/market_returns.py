"""Value-weighted market daily returns from OHLC partitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from config import market_vw_returns_path


def _cap_col(columns: set[str]) -> str:
    if "DlyTotalCap" in columns:
        return "DlyTotalCap"
    return "DlyCap"


def compute_market_vw_returns(ohlc_path: Path) -> pd.DataFrame:
    """One row per trading day: cap-weighted average of DlyRet."""
    ohlc_path = Path(ohlc_path)
    hive = list(ohlc_path.glob("PERMNO=*/*.parquet"))
    if hive:
        dataset = ds.dataset(str(ohlc_path), format="parquet", partitioning="hive")
        schema_cols = set(dataset.schema.names)
        cap_col = _cap_col(schema_cols)
        table = dataset.to_table(columns=["DlyCalDt", "DlyRet", cap_col])
        df = table.to_pandas()
    else:
        df = pd.read_parquet(ohlc_path, columns=["DlyCalDt", "DlyRet", "DlyCap"])
        cap_col = "DlyCap"

    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    cap = df[cap_col].to_numpy(dtype=np.float64)
    ret = df["DlyRet"].to_numpy(dtype=np.float64)
    ok = np.isfinite(ret) & np.isfinite(cap) & (cap > 0.0)
    df = df.loc[ok, ["DlyCalDt", "DlyRet", cap_col]].copy()
    df["_wret"] = df["DlyRet"] * df[cap_col]
    grouped = df.groupby("DlyCalDt", sort=True).agg(
        num=("_wret", "sum"),
        den=(cap_col, "sum"),
    )
    grouped["MktRet"] = (grouped["num"] / grouped["den"]).astype(np.float32)
    out = grouped.reset_index()[["DlyCalDt", "MktRet"]]
    if out.empty:
        raise ValueError(f"empty market return aggregation from {ohlc_path}")
    return out


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
