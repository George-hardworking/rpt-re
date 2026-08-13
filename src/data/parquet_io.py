"""Parquet dataset read helpers for OHLC daily data."""

from pathlib import Path

import numpy as np
import pandas as pd

from data.calendar import market_calendar


def load_calendar(parquet_path: Path) -> pd.DatetimeIndex:
    dates = pd.read_parquet(parquet_path, columns=["DlyCalDt"])["DlyCalDt"]
    return market_calendar(dates)


def permno_list(parquet_path: Path) -> np.ndarray:
    root = Path(parquet_path)
    partition_dirs = sorted(root.glob("PERMNO=*"))
    if partition_dirs:
        permnos = [int(p.name.split("=", 1)[1]) for p in partition_dirs]
        return np.array(sorted(permnos), dtype=np.int64)

    permnos = pd.read_parquet(parquet_path, columns=["PERMNO"])["PERMNO"].unique()
    return np.sort(permnos)


def read_stock(parquet_path: Path, permno: int) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, filters=[("PERMNO", "==", permno)])
    return df.sort_values("DlyCalDt").drop_duplicates(subset=["DlyCalDt"], keep="last")
