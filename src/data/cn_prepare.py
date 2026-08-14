"""China dsf -> PERMNO-partitioned OHLC parquet + PIT universe sidecar."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config import (
    CN_DSF_PATH,
    CN_OHLC_HISTORY_START,
    MARKET_CN,
    market_processed_dir,
    market_sample_config,
)
from data.calendar import market_calendar
from data.cn_universe import filter_panel_to_universe, secu_code_to_permno
from data.parquet_io import ohlc_calendar_path, write_ohlc_calendar
from utils.checkpoint import mark_ohlc_complete, ohlc_is_complete

CN_UNIVERSE_PIT = "universe_pit.parquet"

DSF_USECOLS: tuple[str, ...] = (
    "SecuCode",
    "date",
    "oprc",
    "high",
    "low",
    "prc",
    "svol",
    "ret",
    "float",
    "totcap",
    "indname",
    "chiname",
    "indcode_2",
    "chiname_2",
)

OHLC_OUT_COLS = [
    "PERMNO",
    "DlyCalDt",
    "DlyOpen",
    "DlyHigh",
    "DlyLow",
    "DlyClose",
    "DlyVol",
    "DlyRet",
    "DlyCap",
    "DlyFloatCap",
    "DlyTotalCap",
]


def universe_pit_path(processed_root: Path | None = None) -> Path:
    root = processed_root if processed_root is not None else market_processed_dir(MARKET_CN)
    return root / CN_UNIVERSE_PIT


def _map_dsf_to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("float", "totcap") if c not in df.columns]
    if missing:
        raise KeyError(f"dsf missing cap columns: {missing}")
    out = pd.DataFrame()
    out["PERMNO"] = df["SecuCode"].map(secu_code_to_permno).astype("int64")
    out["DlyCalDt"] = pd.to_datetime(df["date"])
    out["DlyOpen"] = df["oprc"].astype("float64")
    out["DlyHigh"] = df["high"].astype("float64")
    out["DlyLow"] = df["low"].astype("float64")
    out["DlyClose"] = df["prc"].astype("float64")
    out["DlyVol"] = df["svol"].astype("float64")
    out["DlyRet"] = df["ret"].astype("float64")
    out["DlyCap"] = df["float"].astype("float64")
    out["DlyFloatCap"] = df["float"].astype("float64")
    out["DlyTotalCap"] = df["totcap"].astype("float64")
    return out


def _write_permno_partitions(df: pd.DataFrame, output_path: Path, log=print) -> int:
    output_path.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    for permno, grp in df.groupby("PERMNO", sort=True):
        permno = int(permno)
        part = output_path / f"PERMNO={permno}"
        part.mkdir(parents=True, exist_ok=True)
        out_path = part / "part-0.parquet"
        grp[OHLC_OUT_COLS].sort_values("DlyCalDt").to_parquet(out_path, index=False)
        n_rows += len(grp)
        if n_rows % 500_000 == 0:
            log(f"write ohlc rows={n_rows:,}")
    return n_rows


def prepare_cn_ohlc_parquet(
    *,
    dsf_path: Path = CN_DSF_PATH,
    univ_path: Path | None = None,
    output_path: Path | None = None,
    fresh: bool = False,
    log=print,
) -> Path:
    """dsf -> hive OHLC (full history per eligible stock) + universe_pit sidecar."""
    if dsf_path is None:
        dsf_path = CN_DSF_PATH
    dsf_path = Path(dsf_path)
    if not dsf_path.is_file():
        raise FileNotFoundError(f"China dsf not found: {dsf_path}")

    processed_root = market_processed_dir(MARKET_CN)
    if output_path is None:
        output_path = processed_root / "ohlc_daily"
    sample = market_sample_config(MARKET_CN)
    history_start = pd.Timestamp(CN_OHLC_HISTORY_START)
    sample_end = pd.Timestamp(sample.sample_end)

    if not fresh and ohlc_is_complete(output_path):
        log(f"skip cn ohlc (checkpoint): {output_path}")
        pit = universe_pit_path(processed_root)
        if not pit.is_file():
            raise FileNotFoundError(f"missing {pit}; rerun with --fresh")
        if not ohlc_calendar_path(output_path).is_file():
            from data.parquet_io import load_calendar

            load_calendar(output_path, log=log)
        return output_path

    if output_path.exists():
        shutil.rmtree(output_path)
    cal_path = ohlc_calendar_path(output_path)
    if cal_path.exists():
        cal_path.unlink()
    pit_path = universe_pit_path(processed_root)
    if pit_path.exists():
        pit_path.unlink()

    log(f"load dsf {dsf_path}")
    raw = pd.read_parquet(dsf_path, columns=list(DSF_USECOLS))
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.loc[raw["date"] <= sample_end].copy()
    raw["stock_code"] = raw["SecuCode"].astype(str).str.zfill(6)

    log("apply universe filter for PIT membership")
    filtered = filter_panel_to_universe(raw, univ_path=univ_path)
    eligible_permnos = sorted(filtered["stock_code"].map(secu_code_to_permno).unique())
    log(f"eligible stocks after universe={len(eligible_permnos)}")

    pit = filtered[["stock_code", "date"]].copy()
    pit["PERMNO"] = pit["stock_code"].map(secu_code_to_permno).astype("int64")
    pit = pit.rename(columns={"date": "DlyCalDt"})
    pit = pit[["PERMNO", "DlyCalDt"]].drop_duplicates()
    processed_root.mkdir(parents=True, exist_ok=True)
    pit.to_parquet(pit_path, index=False)
    log(f"universe PIT rows={len(pit):,} -> {pit_path}")

    full = raw.loc[
        raw["stock_code"].map(secu_code_to_permno).isin(eligible_permnos)
        & (raw["date"] >= history_start)
        & (raw["date"] <= sample_end)
    ].copy()
    ohlc = _map_dsf_to_ohlc(full)
    log(f"write ohlc partitions stocks={ohlc['PERMNO'].nunique()} rows={len(ohlc):,}")
    n_rows = _write_permno_partitions(ohlc, output_path, log=log)

    calendar = market_calendar(ohlc["DlyCalDt"])
    written = write_ohlc_calendar(output_path, calendar)
    log(
        f"calendar dates={len(calendar)} [{calendar[0].date()}, {calendar[-1].date()}] -> {written}"
    )
    mark_ohlc_complete(output_path)
    log(f"prepared cn ohlc n_rows={n_rows:,} -> {output_path}")
    return output_path


def filter_stock_to_pit(stock_df: pd.DataFrame, pit: pd.DataFrame, permno: int) -> pd.DataFrame:
    keys = pit.loc[pit["PERMNO"] == permno, "DlyCalDt"]
    if keys.empty:
        return stock_df.iloc[0:0]
    allowed = pd.DatetimeIndex(keys)
    out = stock_df.loc[stock_df["DlyCalDt"].isin(allowed)].copy()
    return out.sort_values("DlyCalDt")


def load_universe_pit(processed_root: Path | None = None) -> pd.DataFrame:
    path = universe_pit_path(processed_root)
    if not path.is_file():
        raise FileNotFoundError(f"missing universe PIT: {path}; run cn ohlc first")
    pit = pd.read_parquet(path)
    pit["DlyCalDt"] = pd.to_datetime(pit["DlyCalDt"])
    return pit
