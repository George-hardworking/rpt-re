"""Raw data listing and lightweight CSV inspection."""

from pathlib import Path

import pandas as pd

from rpt_re.config import DATA_ROOT

DEFAULT_CHUNKSIZE = 500_000

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


def list_data_files(subdir: str = "raw") -> list[Path]:
    root = DATA_ROOT / subdir
    return sorted(p for p in root.rglob("*") if p.is_file())


def read_csv_head(path: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, dtype=OHLC_DTYPES)


def read_csv_sample(path: Path, nrows: int = 10_000) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, dtype=OHLC_DTYPES)


def csv_row_count(path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> int:
    return sum(
        len(chunk)
        for chunk in pd.read_csv(
            path,
            chunksize=chunksize,
            usecols=["PERMNO"],
            dtype={"PERMNO": "int64"},
        )
    )


def missing_value_stats(path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    total_rows = 0
    null_counts = None
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=OHLC_DTYPES):
        total_rows += len(chunk)
        chunk_nulls = chunk.isna().sum()
        null_counts = chunk_nulls if null_counts is None else null_counts + chunk_nulls

    stats = pd.DataFrame(
        {
            "null_count": null_counts,
            "null_pct": null_counts / total_rows * 100,
        }
    )
    return stats.sort_values("null_count", ascending=False)
