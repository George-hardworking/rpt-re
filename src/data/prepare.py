"""Convert raw CRSP OHLC CSV to partitioned parquet without filling missing values."""

import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import CSV_CHUNKSIZE, OHLC_PARQUET, PROCESSED_DIR, RAW_OHLC_CSV

OHLC_DTYPES: dict[str, str] = {
    "PERMNO": "int64",
    "HdrCUSIP": "string",
    "Ticker": "string",
    "PERMCO": "int64",
    "DlyCalDt": "string",
    "DlyCap": "float64",
    "DlyRet": "float64",
    "DlyRetx": "float64",
    "DlyVol": "float64",
    "DlyClose": "float64",
    "DlyLow": "float64",
    "DlyHigh": "float64",
    "DlyOpen": "float64",
}

PREPARE_COLS = [
    "PERMNO",
    "DlyCalDt",
    "DlyCap",
    "DlyRet",
    "DlyVol",
    "DlyClose",
    "DlyLow",
    "DlyHigh",
    "DlyOpen",
]


def prepare_ohlc_parquet(
    raw_path: Path = RAW_OHLC_CSV,
    output_path: Path = OHLC_PARQUET,
    chunksize: int = CSV_CHUNKSIZE,
) -> Path:
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw OHLC CSV not found: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=OHLC_DTYPES):
        chunk = chunk[PREPARE_COLS].copy()
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=["PERMNO"],
        )

    return output_path
