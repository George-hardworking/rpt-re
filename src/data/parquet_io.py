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

CALENDAR_SCAN_EVERY = 1000


def ohlc_calendar_path(parquet_path: Path) -> Path:
    """Sidecar next to the hive dataset, not inside it (avoids schema mix-in)."""
    return Path(parquet_path).resolve().parent / "ohlc_calendar.parquet"


def write_ohlc_calendar(parquet_path: Path, calendar: pd.DatetimeIndex) -> Path:
    path = ohlc_calendar_path(parquet_path)
    pd.DataFrame({"DlyCalDt": pd.DatetimeIndex(calendar)}).to_parquet(path, index=False)
    return path


def _as_datetime64_ns(values) -> np.ndarray:
    return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]")


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
        if not ohlc_calendar_path(output_path).is_file():
            load_calendar(output_path, log=log)
        return output_path

    if output_path.exists():
        shutil.rmtree(output_path)
    cal_path = ohlc_calendar_path(output_path)
    if cal_path.exists():
        cal_path.unlink()
    output_path.mkdir(parents=True)

    n_rows = 0
    n_chunks = 0
    cal_acc = np.array([], dtype="datetime64[ns]")
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=OHLC_DTYPES):
        n_chunks += 1
        n_rows += len(chunk)
        chunk = chunk[PREPARE_COLS].copy()
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
        cal_acc = np.union1d(cal_acc, _as_datetime64_ns(chunk["DlyCalDt"].unique()))
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=["PERMNO"],
        )
        log(f"prepare chunk={n_chunks} rows={n_rows:,} -> {output_path}")

    calendar = market_calendar(pd.Series(cal_acc))
    written = write_ohlc_calendar(output_path, calendar)
    log(f"calendar dates={len(calendar)} [{calendar[0].date()}, {calendar[-1].date()}] -> {written}")
    mark_ohlc_complete(output_path)
    return output_path


def _scan_hive_calendar(parquet_path: Path, log=None) -> pd.DatetimeIndex:
    files = sorted(Path(parquet_path).glob("PERMNO=*/*.parquet"))
    n = len(files)
    acc = np.array([], dtype="datetime64[ns]")
    for i, path in enumerate(files, 1):
        col = pq.ParquetFile(path).read(columns=["DlyCalDt"]).column(0).to_numpy()
        acc = np.union1d(acc, _as_datetime64_ns(col))
        if log is not None and (i % CALENDAR_SCAN_EVERY == 0 or i == n):
            log(f"calendar scan {i}/{n} unique_dates={len(acc)}")
    return market_calendar(pd.Series(acc))


def load_calendar(parquet_path: Path, log=None) -> pd.DatetimeIndex:
    parquet_path = Path(parquet_path)
    cache = ohlc_calendar_path(parquet_path)
    if cache.is_file():
        if log is not None:
            log(f"calendar cache: {cache}")
        return market_calendar(pd.read_parquet(cache)["DlyCalDt"])

    if log is not None:
        log(f"building calendar from {parquet_path}")
    hive_files = list(parquet_path.glob("PERMNO=*/*.parquet"))
    if hive_files:
        idx = _scan_hive_calendar(parquet_path, log=log)
    else:
        dates = pd.read_parquet(parquet_path, columns=["DlyCalDt"])["DlyCalDt"]
        idx = market_calendar(dates)
    assert len(idx) > 0, f"empty calendar from {parquet_path}"
    write_ohlc_calendar(parquet_path, idx)
    if log is not None:
        log(f"calendar dates={len(idx)} [{idx[0].date()}, {idx[-1].date()}] -> {cache}")
    return idx


def permno_list(parquet_path: Path) -> np.ndarray:
    root = Path(parquet_path)
    partition_dirs = sorted(root.glob("PERMNO=*"))
    if partition_dirs:
        permnos = [int(p.name.split("=", 1)[1]) for p in partition_dirs]
        return np.array(sorted(permnos), dtype=np.int64)

    permnos = pd.read_parquet(parquet_path, columns=["PERMNO"])["PERMNO"].unique()
    return np.sort(permnos)


def _read_permno_partition(root: Path, permno: int) -> pd.DataFrame:
    """Read one hive partition by path. Does not scan sibling PERMNO=* directories."""
    part = Path(root) / f"PERMNO={permno}"
    files = sorted(part.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet in {part}")
    tables = [pq.ParquetFile(path).read() for path in files]
    df = pa.concat_tables(tables).to_pandas()
    if "PERMNO" not in df.columns:
        df.insert(0, "PERMNO", np.int64(permno))
    return df


def read_stock(parquet_path: Path, permno: int) -> pd.DataFrame:
    df = _read_permno_partition(parquet_path, permno)
    return df.sort_values("DlyCalDt").drop_duplicates(subset=["DlyCalDt"], keep="last")


def read_stock_features(features_path: Path, permno: int) -> pd.DataFrame:
    df = _read_permno_partition(features_path, permno)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df.sort_values("DlyCalDt").drop_duplicates(subset=["DlyCalDt"], keep="last")
    return df.set_index("DlyCalDt")


def read_feature_row(features_path: Path, permno: int, as_of: pd.Timestamp) -> dict:
    as_of = pd.Timestamp(as_of)
    panel = read_stock_features(features_path, permno)
    if as_of not in panel.index:
        raise KeyError(f"no feature row PERMNO={permno} DlyCalDt={as_of}")
    rec = panel.loc[as_of].to_dict()
    rec["PERMNO"] = permno
    rec["DlyCalDt"] = as_of
    return rec
