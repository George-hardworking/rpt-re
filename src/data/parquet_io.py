"""Parquet dataset read helpers for OHLC daily data."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CSV_CHUNKSIZE, OHLC_DTYPES, OHLC_PARQUET, PREPARE_COLS, PROCESSED_DIR, RAW_OHLC_CSV
from data.calendar import market_calendar
from utils.checkpoint import mark_ohlc_complete, ohlc_is_complete


def prepare_ohlc_parquet(
    raw_path: Path = RAW_OHLC_CSV,
    output_path: Path = OHLC_PARQUET,
    chunksize: int = CSV_CHUNKSIZE,
    *,
    fresh: bool = False,
    log=print,
) -> Path:
    """Convert raw CRSP OHLC CSV to PERMNO-partitioned parquet without filling missing values."""
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw OHLC CSV not found: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not fresh and ohlc_is_complete(output_path):
        log(f"skip ohlc (checkpoint): {output_path}")
        return output_path

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    n_rows = 0
    n_chunks = 0
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=OHLC_DTYPES):
        n_chunks += 1
        n_rows += len(chunk)
        chunk = chunk[PREPARE_COLS].copy()
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=["PERMNO"],
        )
        log(f"prepare chunk={n_chunks} rows={n_rows:,} -> {output_path}")

    mark_ohlc_complete(output_path)
    return output_path


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


def read_stock_features(features_path: Path, permno: int) -> pd.DataFrame:
    df = pd.read_parquet(features_path, filters=[("PERMNO", "==", permno)])
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df.sort_values("DlyCalDt").drop_duplicates(subset=["DlyCalDt"], keep="last")
    return df.set_index("DlyCalDt")


def read_feature_row(features_path: Path, permno: int, as_of: pd.Timestamp) -> dict:
    as_of = pd.Timestamp(as_of)
    df = pd.read_parquet(
        features_path,
        filters=[("PERMNO", "==", permno), ("DlyCalDt", "==", as_of)],
    )
    if df.empty:
        raise KeyError(f"no feature row PERMNO={permno} DlyCalDt={as_of}")
    return df.iloc[0].to_dict()
