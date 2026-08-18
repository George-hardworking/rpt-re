"""01 — weekly R5 portfolio performance (paper Table I layout).

Read-only: loads existing step-04 all_h1.xlsx and step-05 h1.xlsx (no backtest).
US: equal + total cap weight. CN: equal + float cap + total cap weight.

Run (from repo root, 5020_env):
  python "outputs/analysis_results/01_short-horizon portfolio performance/01_portfolio_weekly_h1.py"
  python "outputs/analysis_results/01_short-horizon portfolio performance/01_portfolio_weekly_h1.py" --market cn
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from analysis.table_format import format_cell, format_value_only
from config import (
    BACKTEST_BENCHMARK_ROOT,
    BACKTEST_CNN_ROOT,
    MARKET_CN,
    MARKET_US,
)

HERE = Path(__file__).resolve().parent
STEM = "01_portfolio_weekly_h1"
HORIZON = 5
FREQ_DIR = "weekly"

CNN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("I5/R5", "I5_R5"),
    ("I20/R5", "I20_R5"),
    ("I60/R5", "I60_R5"),
)

BENCHMARK_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("MOM/R5", "MOM", "mom"),
    ("STR/R5", "REV1m_STR", "rev1m_str"),
    ("WSTR/R5", "REV1w_WSTR", "rev1w_wstr"),
    ("TREND/R5", "TREND_HZZ", "trend_hzz"),
)

DECILE_SIG_RANKS: tuple[tuple[str, str], ...] = (
    ("Low", "D01"),
    ("2", "D02"),
    ("3", "D03"),
    ("4", "D04"),
    ("5", "D05"),
    ("6", "D06"),
    ("7", "D07"),
    ("8", "D08"),
    ("9", "D09"),
    ("High", "D10"),
    ("H-L", "DH"),
)

RET_METRIC = "Annualized Return"
SR_METRIC = "Sharpe Ratio"
TO_METRIC = "Turnover (annualized)"

US_PANELS: tuple[tuple[str, str, str], ...] = (
    ("Equal-Weight", "equal", "Equal-weight (EW): 1/N within each decile portfolio."),
    (
        "TotalCap-Weight",
        "total",
        "Total market-cap weight at formation (US: MarketCap).",
    ),
)

CN_PANELS: tuple[tuple[str, str, str], ...] = (
    ("Equal-Weight", "equal", "Equal-weight (EW): 1/N within each decile portfolio."),
    (
        "FloatCap-Weight",
        "float",
        "Float market-cap weight at formation (CN: FloatCap / 流通市值).",
    ),
    (
        "TotalCap-Weight",
        "total",
        "Total market-cap weight at formation (CN: TotalCap / 总市值).",
    ),
)

NOTES_LINES: tuple[str, ...] = (
    "Weekly R5 portfolio summary from step-04 CNN all_h1.xlsx and step-05 benchmark h1.xlsx.",
    f"Return column: {RET_METRIC} (raw portfolio return, annualized). "
    "NOT Annualized Active Return.",
    f"Sharpe column: {SR_METRIC} = Annualized Return / Annualized Risk (both raw).",
    f"Turnover row: {TO_METRIC} on H-L (DH) portfolio.",
    "H-L return stars use Newey-West t-stat on the long-short spread.",
)
_T_RE = re.compile(r"^([-+]?\d*\.?\d+)(\**)$")


def log(msg: str) -> None:
    print(msg, flush=True)


def panels_for_market(market: str) -> tuple[tuple[str, str, str], ...]:
    if market == MARKET_CN:
        return CN_PANELS
    if market == MARKET_US:
        return US_PANELS
    raise ValueError(f"unsupported market={market}")


def parse_t_stat(cell: object) -> float:
    if cell is None or (isinstance(cell, float) and not np.isfinite(cell)):
        return float("nan")
    text = str(cell).strip()
    if not text:
        return float("nan")
    m = _T_RE.match(text)
    if not m:
        return float("nan")
    return float(m.group(1))


def read_h1_models(path: Path, sheet: str) -> dict[str, pd.Series]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = pd.read_excel(path, sheet_name=sheet, header=[0, 1])
    name_col = ("metric", "sig_rank")
    models: dict[str, pd.Series] = {}
    for _, row in raw.iterrows():
        model = row[name_col]
        if pd.isna(model) or str(model) in {"sig_rank", "nan"}:
            continue
        models[str(model)] = row
    if not models:
        raise ValueError(f"no model rows in {path} sheet={sheet}")
    return models


def cnn_h1_path(market: str) -> Path:
    path = BACKTEST_CNN_ROOT / market / FREQ_DIR / "all_h1.xlsx"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run ./scripts/sh/04_backtest.sh --market {market}"
        )
    return path


def benchmark_h1_path(market: str, signal_dir: str) -> Path:
    path = BACKTEST_BENCHMARK_ROOT / signal_dir / market / FREQ_DIR / "h1.xlsx"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run ./scripts/sh/05_benchmark_signals.sh backtest "
            f"--market {market} --horizons {HORIZON}"
        )
    return path


def collect_sources(market: str) -> dict[str, tuple[Path, str]]:
    sources: dict[str, tuple[Path, str]] = {}
    cnn_path = cnn_h1_path(market)
    for label, row_name in CNN_COLUMNS:
        sources[label] = (cnn_path, row_name)
    for label, signal_col, signal_dir in BENCHMARK_COLUMNS:
        sources[label] = (benchmark_h1_path(market, signal_dir), signal_col)
    return sources


def load_panel(market: str, sheet: str) -> dict[str, pd.Series]:
    sources = collect_sources(market)
    cache: dict[tuple[Path, str], dict[str, pd.Series]] = {}
    panel: dict[str, pd.Series] = {}
    for label, (path, row_name) in sources.items():
        key = (path, sheet)
        if key not in cache:
            cache[key] = read_h1_models(path, sheet)
        models = cache[key]
        if row_name not in models:
            raise KeyError(f"{path} sheet={sheet} missing row {row_name!r} for column {label}")
        panel[label] = models[row_name]
    return panel


def _num(row: pd.Series, metric: str, sig_rank: str) -> float:
    val = row[(metric, sig_rank)]
    if pd.isna(val):
        return float("nan")
    return float(val)


def _metric_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        out.append(f"{col} Ret (ann.)")
        out.append(f"{col} SR")
    return out


def build_panel_block(
    market: str,
    panel_name: str,
    sheet: str,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = _metric_columns(columns)
    frame_cols = ["Weight scheme", "Row", *metric_cols]
    panel = load_panel(market, sheet)

    display_rows: list[dict[str, object]] = []
    raw_ret_rows: list[dict[str, object]] = []
    raw_sr_rows: list[dict[str, object]] = []
    raw_to_rows: list[dict[str, object]] = []

    for row_label, sig_rank in DECILE_SIG_RANKS:
        disp_row: dict[str, object] = {"Weight scheme": panel_name, "Row": row_label}
        raw_ret_row: dict[str, object] = {"Weight scheme": panel_name, "Row": row_label}
        raw_sr_row: dict[str, object] = {"Weight scheme": panel_name, "Row": row_label}
        for col in columns:
            row = panel[col]
            ret = _num(row, RET_METRIC, sig_rank)
            sr = _num(row, SR_METRIC, sig_rank)
            ret_key = f"{col} Ret (ann.)"
            sr_key = f"{col} SR"
            raw_ret_row[col] = ret
            raw_sr_row[col] = sr
            if sig_rank == "DH":
                t = parse_t_stat(row[(RET_METRIC, "t")])
                disp_row[ret_key] = format_cell(ret, t, value_decimals=2, t_decimals=2)
            else:
                disp_row[ret_key] = format_value_only(ret, decimals=2)
            disp_row[sr_key] = format_value_only(sr, decimals=2)
        display_rows.append(disp_row)
        raw_ret_rows.append(raw_ret_row)
        raw_sr_rows.append(raw_sr_row)

    to_label = "Turnover"
    disp_to: dict[str, object] = {"Weight scheme": panel_name, "Row": to_label}
    raw_to_row: dict[str, object] = {"Weight scheme": panel_name, "Row": to_label}
    for col in columns:
        row = panel[col]
        to_val = _num(row, TO_METRIC, "DH")
        ret_key = f"{col} Ret (ann.)"
        sr_key = f"{col} SR"
        raw_to_row[col] = to_val
        disp_to[ret_key] = f"{to_val:.0f}%" if np.isfinite(to_val) else ""
        disp_to[sr_key] = ""
    display_rows.append(disp_to)
    raw_ret_rows.append(raw_to_row)
    raw_to_rows.append(raw_to_row)

    display = pd.DataFrame(display_rows, columns=frame_cols)
    raw_ret = pd.DataFrame(raw_ret_rows, columns=["Weight scheme", "Row", *columns])
    raw_sr = pd.DataFrame(raw_sr_rows, columns=["Weight scheme", "Row", *columns])
    raw_to = pd.DataFrame(raw_to_rows, columns=["Weight scheme", "Row", *columns])
    return display, raw_ret, raw_sr, raw_to


def build_tables(market: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [c[0] for c in CNN_COLUMNS] + [c[0] for c in BENCHMARK_COLUMNS]
    display_parts: list[pd.DataFrame] = []
    raw_ret_parts: list[pd.DataFrame] = []
    raw_sr_parts: list[pd.DataFrame] = []
    raw_to_parts: list[pd.DataFrame] = []

    for panel_name, sheet, _desc in panels_for_market(market):
        disp, raw_ret, raw_sr, raw_to = build_panel_block(market, panel_name, sheet, columns)
        display_parts.append(disp)
        raw_ret_parts.append(raw_ret)
        raw_sr_parts.append(raw_sr)
        raw_to_parts.append(raw_to)

    return (
        pd.concat(display_parts, axis=0, ignore_index=True),
        pd.concat(raw_ret_parts, axis=0, ignore_index=True),
        pd.concat(raw_sr_parts, axis=0, ignore_index=True),
        pd.concat(raw_to_parts, axis=0, ignore_index=True),
    )


def build_notes(market: str) -> pd.DataFrame:
    rows = [{"item": "market", "value": market}]
    for panel_name, sheet, desc in panels_for_market(market):
        rows.append({"item": f"panel: {panel_name}", "value": f"source sheet={sheet!r}; {desc}"})
    for line in NOTES_LINES:
        rows.append({"item": "note", "value": line})
    return pd.DataFrame(rows)


def output_path(market: str) -> Path:
    return HERE / f"{STEM}_{market}.xlsx"


def write_excel(
    market: str,
    display: pd.DataFrame,
    raw_returns: pd.DataFrame,
    raw_sharpe: pd.DataFrame,
    raw_turnover: pd.DataFrame,
    notes: pd.DataFrame,
) -> Path:
    out = output_path(market)
    tmp = out.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="notes", index=False)
        display.to_excel(writer, sheet_name="display", index=False)
        raw_returns.to_excel(writer, sheet_name="raw_returns", index=False)
        raw_sharpe.to_excel(writer, sheet_name="raw_sharpe", index=False)
        raw_turnover.to_excel(writer, sheet_name="raw_turnover", index=False)
    tmp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="01 portfolio weekly H1 summary")
    parser.add_argument("--market", choices=(MARKET_US, MARKET_CN), default=MARKET_US)
    args = parser.parse_args()

    display, raw_ret, raw_sr, raw_to = build_tables(args.market)
    notes = build_notes(args.market)
    out = write_excel(args.market, display, raw_ret, raw_sr, raw_to, notes)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
