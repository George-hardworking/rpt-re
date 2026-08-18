"""Audit akshare SPY / 510300 coverage for US/CN backtest test periods."""

from __future__ import annotations

from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

US_START, US_END = "2001-01-01", "2019-12-31"
CN_START, CN_END = "2015-01-01", "2019-12-31"


def load_cal(path: Path, start: str, end: str) -> pd.DatetimeIndex:
    df = pd.read_parquet(path)
    col = "DlyCalDt" if "DlyCalDt" in df.columns else df.columns[0]
    d = pd.to_datetime(df[col]).sort_values()
    return pd.DatetimeIndex(d[(d >= start) & (d <= end)].unique())


def audit(
    name: str,
    df: pd.DataFrame,
    start: str,
    end: str,
    *,
    cal: pd.DatetimeIndex | None = None,
    price_col: str = "close",
) -> None:
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    dates = pd.DatetimeIndex(sub["date"].unique()).sort_values()
    print(f"\n{'=' * 60}")
    print(name)
    print(f"period: {start} ~ {end}")
    print("columns:", list(sub.columns))
    vol_like = [
        c
        for c in sub.columns
        if any(k in str(c).lower() for k in ["vol", "波动", "振幅", "amplitude", "std"])
    ]
    print(
        "volatility-like columns:",
        vol_like if vol_like else "NONE (compute from price returns)",
    )
    print(
        "rows:",
        len(sub),
        "unique dates:",
        len(dates),
        "range:",
        dates.min().date(),
        dates.max().date(),
    )

    for c in [price_col, "open", "high", "low", "volume"]:
        if c in sub.columns:
            na = int(sub[c].isna().sum())
            print(f"  missing {c}: {na} ({na / len(sub):.2%})")

    if cal is not None:
        have = set(dates)
        expected = set(cal)
        missing = sorted(expected - have)
        extra = sorted(have - expected)
        print(f"  vs project calendar: expected {len(expected)} trading days")
        print(
            f"  covered {len(have & expected)} | missing {len(missing)} | extra non-cal {len(extra)}"
        )
        if missing:
            print("  first missing cal dates:", [d.date() for d in missing[:8]])
            print("  last missing cal dates:", [d.date() for d in missing[-8:]])

    sub = sub.sort_values("date")
    px = sub[price_col].astype(float)
    ret = px.pct_change().replace([np.inf, -np.inf], np.nan)
    print(
        f"  daily return non-null: {int(ret.notna().sum())} | missing after pct_change: {int(ret.isna().sum())}"
    )
    ann_vol = float(ret.std() * np.sqrt(252))
    print(f"  full-period daily-return annualized vol (Figure5 scaling): {ann_vol:.4f}")

    wk_df = sub.set_index("date")
    wk = wk_df[price_col].astype(float).pct_change().resample("W-FRI").apply(
        lambda x: (1.0 + x).prod() - 1.0
    )
    print(
        f"  weekly returns (W-FRI): {int(wk.notna().sum())} weeks | ann vol ~ {float(wk.std() * np.sqrt(52)):.4f}"
    )


def main() -> None:
    spy = ak.stock_us_daily(symbol="SPY", adjust="")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date")

    cn = ak.fund_etf_hist_sina(symbol="sh510300")
    cn["date"] = pd.to_datetime(cn["date"])
    cn = cn.sort_values("date")

    print("=== akshare spot/info interfaces for 510300 vol fields ===")
    try:
        spot = ak.fund_etf_spot_em()
        print("fund_etf_spot_em cols:", list(spot.columns))
        mask = spot.astype(str).apply(lambda s: s.str.contains("510300", na=False)).any(axis=1)
        if mask.any():
            print("510300 spot row:\n", spot.loc[mask].head(1).T)
    except Exception as e:
        print("fund_etf_spot_em fail:", e)

    try:
        info = ak.fund_etf_fund_info_em(fund="510300")
        if isinstance(info, pd.DataFrame):
            print("fund_etf_fund_info_em cols:", list(info.columns))
            print(info.head(3))
        else:
            print("fund_etf_fund_info_em returned:", type(info))
    except Exception as e:
        print("fund_etf_fund_info_em fail:", e)

    us_cal = load_cal(
        Path("/data/kaibiao/data/projects/rpt-re/processed/us/ohlc_calendar.parquet"),
        US_START,
        US_END,
    )
    cn_cal = load_cal(
        Path("/data/kaibiao/data/projects/rpt-re/processed/cn/ohlc_calendar.parquet"),
        CN_START,
        CN_END,
    )

    audit("SPY (ak.stock_us_daily)", spy, US_START, US_END, cal=us_cal)
    audit("CSI300 ETF 510300 (ak.fund_etf_hist_sina)", cn, CN_START, CN_END, cal=cn_cal)

    print("\n510300 listing date:", cn["date"].min().date())
    print(
        "CN test covers full listing history in period:",
        cn["date"].min() <= pd.Timestamp(CN_START),
    )


if __name__ == "__main__":
    main()
