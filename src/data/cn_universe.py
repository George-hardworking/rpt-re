"""China A-share daily universe: allstk membership + finance/utilities/B-share exclusions.

Rules aligned with similarity-research p12 / weekfreq (no cross-repo import).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from config import CN_UNIV_PATH

B_SHARE_PREFIXES = ("200", "900")
ST_ABBR_RE = re.compile(r"ST|\*ST|退|退市", re.IGNORECASE)

FINANCE_INDNAME = frozenset({"Banking", "Non-bank Finance", "Financial Services"})
FINANCE_CHINAME = frozenset({"银行", "非银金融", "金融服务"})
UTILITIES_INDNAME = frozenset({"Public Utilities"})
UTILITIES_CHINAME = frozenset({"公用事业"})

OPTIONAL_MARKET_COLS = ("Markettype", "MarketType")

DSF_UNIVERSE_COLS: tuple[str, ...] = (
    "SecuCode",
    "date",
    "indname",
    "chiname",
    "indcode_2",
    "chiname_2",
)


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return series.map(lambda x: str(x).strip().zfill(6)).astype("string")


def secu_code_to_permno(code: str | int) -> int:
    return int(str(code).strip())


def permno_to_secu_code(permno: int) -> str:
    return str(int(permno)).zfill(6)


def is_b_share_code(code: str) -> bool:
    return str(code).zfill(6).startswith(B_SHARE_PREFIXES)


def is_st_abbrev(name: object) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    return bool(ST_ABBR_RE.search(name))


def _series_is_finance_industry(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for col, labels in (
        ("indname", FINANCE_INDNAME),
        ("chiname", FINANCE_CHINAME),
        ("chiname_2", FINANCE_CHINAME),
    ):
        if col in df.columns:
            out = out | df[col].astype(str).isin(labels)
    if "indcode_2" in df.columns:
        codes = df["indcode_2"].astype(str)
        out = out | codes.str.contains("SW480", na=False) | codes.str.contains("SW490", na=False)
    return out


def _series_is_utilities_industry(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for col, labels in (("indname", UTILITIES_INDNAME), ("chiname", UTILITIES_CHINAME)):
        if col in df.columns:
            out = out | df[col].astype(str).isin(labels)
    if "indcode_2" in df.columns:
        out = out | df["indcode_2"].astype(str).str.contains("SW410", na=False)
    return out


def universe_filter_mask(df: pd.DataFrame) -> pd.Series:
    """True = keep row (passes finance / utilities / B-share exclusions)."""
    codes = normalize_stock_code(
        df.get("SecuCode", df.get("stock_code", pd.Series(dtype=str)))
    )
    b_share = codes.map(is_b_share_code)
    st = df["SecuAbbr"].map(is_st_abbrev) if "SecuAbbr" in df.columns else pd.Series(False, index=df.index)
    finance = _series_is_finance_industry(df)
    utilities = _series_is_utilities_industry(df)
    for mcol in OPTIONAL_MARKET_COLS:
        if mcol in df.columns:
            b_share = b_share | df[mcol].isin([2, 8])
    return ~(b_share | st | finance | utilities)


def filter_dsf_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    mask = universe_filter_mask(df)
    return df.loc[mask].copy()


@lru_cache(maxsize=4)
def _load_univ_membership_cached(
    univ_path: str,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    df = pd.read_parquet(univ_path, columns=["date", "SecuCode"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["SecuCode"] = normalize_stock_code(df["SecuCode"])
    if date_start:
        df = df.loc[df["date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df.loc[df["date"] <= pd.Timestamp(date_end)]
    return df.drop_duplicates(subset=["date", "SecuCode"]).reset_index(drop=True)


def load_univ_membership(
    *,
    univ_path: str | Path | None = None,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    path = Path(univ_path) if univ_path is not None else CN_UNIV_PATH
    if not path.is_file():
        raise FileNotFoundError(f"China universe file not found: {path}")
    start = "" if date_start is None else pd.Timestamp(date_start).strftime("%Y-%m-%d")
    end = "" if date_end is None else pd.Timestamp(date_end).strftime("%Y-%m-%d")
    return _load_univ_membership_cached(str(path.resolve()), start, end).copy()


def filter_panel_to_universe(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    stock_col: str = "stock_code",
    univ: pd.DataFrame | None = None,
    univ_path: str | Path | None = None,
) -> pd.DataFrame:
    """allstk membership (same day) then finance / utilities / B-share exclusions."""
    if df.empty:
        return df

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    if stock_col in out.columns:
        out[stock_col] = out[stock_col].astype(str).str.zfill(6)
    added_secu = False
    if "SecuCode" not in out.columns:
        if stock_col not in out.columns:
            raise ValueError("panel needs SecuCode or stock_code for universe filter")
        out["SecuCode"] = out[stock_col]
        added_secu = True
    else:
        out["SecuCode"] = normalize_stock_code(out["SecuCode"])

    if univ is None:
        univ = load_univ_membership(
            univ_path=univ_path,
            date_start=out[date_col].min(),
            date_end=out[date_col].max(),
        )
    else:
        univ = univ.copy()
        univ["date"] = pd.to_datetime(univ["date"]).dt.normalize()
        univ["SecuCode"] = normalize_stock_code(univ["SecuCode"])

    membership = univ[["date", "SecuCode"]].drop_duplicates()
    if date_col != "date":
        membership = membership.rename(columns={"date": date_col})
    out = out.merge(membership, on=[date_col, "SecuCode"], how="inner")
    out = filter_dsf_exclusions(out)

    if added_secu and "SecuCode" in out.columns:
        out = out.drop(columns=["SecuCode"])
    return out
