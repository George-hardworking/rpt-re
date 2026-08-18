"""01 — portfolio performance tables (paper Table I / II layout).

Read-only: loads step-04 all_h{k}.xlsx and step-05 h{k}.xlsx (no backtest).
H-L uses H1 (t+1); display rows DH Ret2 / DH Ret3 use H2 / H3 eval horizons.
Turnover: paper monthly turnover (GKX 2020) on H-L (DH), scaled from engine
``Turnover (annualized)`` per Table I / II convention.
US: equal + total cap weight. CN: equal + float cap + total cap weight.

Run (from repo root, 5020_env):
  python "outputs/analysis_results/01_short-horizon portfolio performance/01_portfolio_weekly_h1.py"
  python "outputs/analysis_results/01_short-horizon portfolio performance/01_portfolio_weekly_h1.py" --freq monthly --market cn
  python "outputs/analysis_results/01_short-horizon portfolio performance/01_portfolio_weekly_h1.py" --freq quarterly --market us
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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

DH_EVAL_HORIZON_ROWS: tuple[tuple[str, int], ...] = (
    ("DH Ret2", 2),
    ("DH Ret3", 3),
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

_T_RE = re.compile(r"^([-+]?\d*\.?\d+)(\**)$")


@dataclass(frozen=True)
class FreqSpec:
    name: str
    stem: str
    freq_dir: str
    horizon: int
    paper_table: str
    periods_per_year: int
    holding_months: float

    @property
    def label(self) -> str:
        return f"R{self.horizon}"

    @property
    def cnn_columns(self) -> tuple[tuple[str, str], ...]:
        h = self.horizon
        return (
            (f"I5/R{h}", f"I5_R{h}"),
            (f"I20/R{h}", f"I20_R{h}"),
            (f"I60/R{h}", f"I60_R{h}"),
        )

    @property
    def benchmark_columns(self) -> tuple[tuple[str, str, str], ...]:
        h = self.horizon
        return (
            (f"MOM/R{h}", "MOM", "mom"),
            (f"STR/R{h}", "REV1m_STR", "rev1m_str"),
            (f"WSTR/R{h}", "REV1w_WSTR", "rev1w_wstr"),
            (f"TREND/R{h}", "TREND_HZZ", "trend_hzz"),
        )

    def notes_lines(self) -> tuple[str, ...]:
        return (
            f"{self.name.capitalize()} {self.label} portfolio summary from step-04 "
            f"CNN all_h{{k}}.xlsx and step-05 benchmark h{{k}}.xlsx "
            f"({self.freq_dir}/). Paper reference: {self.paper_table}.",
            f"Return column: {RET_METRIC} (raw portfolio return, annualized). "
            "NOT Annualized Active Return.",
            f"Sharpe column: {SR_METRIC} = Annualized Return / Annualized Risk (both raw).",
            "H-L: eval horizon H1 (signal at t, return over 1st future rebalance period).",
            "DH Ret2 / DH Ret3: H-L (DH) annualized return at eval horizons H2 / H3 "
            "(2nd / 3rd future rebalance period).",
            "Turnover row: monthly turnover on H-L (DH), paper Table I/II / GKX (2020) "
            f"convention (holding ≈ {self.holding_months} month(s); scaled from backtest "
            f"{TO_METRIC}).",
            "H-L and DH Ret2/Ret3 return stars use Newey-West t-stat on the long-short spread.",
        )


FREQ_SPECS: dict[str, FreqSpec] = {
    "weekly": FreqSpec("weekly", "01_portfolio_weekly_h1", "weekly", 5, "Table I", 52, 0.25),
    "monthly": FreqSpec("monthly", "01_portfolio_monthly_h1", "monthly", 20, "Table II (top)", 12, 1.0),
    "quarterly": FreqSpec(
        "quarterly", "01_portfolio_quarterly_h1", "quarterly", 60, "Table II (bottom)", 4, 3.0
    ),
}


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


def cnn_all_path(spec: FreqSpec, market: str, eval_horizon: int) -> Path:
    return BACKTEST_CNN_ROOT / market / spec.freq_dir / f"all_h{eval_horizon}.xlsx"


def cnn_single_path(spec: FreqSpec, market: str, row_name: str, eval_horizon: int) -> Path:
    return BACKTEST_CNN_ROOT / market / spec.freq_dir / f"{row_name}_h{eval_horizon}.xlsx"


def benchmark_path(spec: FreqSpec, market: str, signal_dir: str, eval_horizon: int) -> Path:
    return BACKTEST_BENCHMARK_ROOT / signal_dir / market / spec.freq_dir / f"h{eval_horizon}.xlsx"


def collect_sources(spec: FreqSpec, market: str, eval_horizon: int) -> dict[str, tuple[Path, str]]:
    sources: dict[str, tuple[Path, str]] = {}
    cnn_path = cnn_all_path(spec, market, eval_horizon)
    for label, row_name in spec.cnn_columns:
        sources[label] = (cnn_path, row_name)
    for label, signal_col, signal_dir in spec.benchmark_columns:
        sources[label] = (benchmark_path(spec, market, signal_dir, eval_horizon), signal_col)
    return sources


def _read_models_cached(
    cache: dict[tuple[Path, str], dict[str, pd.Series]],
    path: Path,
    sheet: str,
) -> dict[str, pd.Series]:
    key = (path, sheet)
    if key not in cache:
        if not path.is_file():
            raise FileNotFoundError(path)
        cache[key] = read_h1_models(path, sheet)
    return cache[key]


def _resolve_cnn_row(
    spec: FreqSpec,
    market: str,
    sheet: str,
    eval_horizon: int,
    row_name: str,
    cache: dict[tuple[Path, str], dict[str, pd.Series]],
) -> pd.Series:
    all_path = cnn_all_path(spec, market, eval_horizon)
    if all_path.is_file():
        models = _read_models_cached(cache, all_path, sheet)
        if row_name in models:
            return models[row_name]
    single_path = cnn_single_path(spec, market, row_name, eval_horizon)
    models = _read_models_cached(cache, single_path, sheet)
    if row_name not in models:
        raise KeyError(f"{single_path} sheet={sheet} missing row {row_name!r}")
    return models[row_name]


def load_panel(
    spec: FreqSpec,
    market: str,
    sheet: str,
    eval_horizon: int = 1,
) -> dict[str, pd.Series]:
    sources = collect_sources(spec, market, eval_horizon)
    cache: dict[tuple[Path, str], dict[str, pd.Series]] = {}
    panel: dict[str, pd.Series] = {}
    cnn_labels = {c[0] for c in spec.cnn_columns}
    for label, (path, row_name) in sources.items():
        if label in cnn_labels:
            panel[label] = _resolve_cnn_row(spec, market, sheet, eval_horizon, row_name, cache)
            continue
        models = _read_models_cached(cache, path, sheet)
        if row_name not in models:
            raise KeyError(f"{path} sheet={sheet} missing row {row_name!r} for column {label}")
        panel[label] = models[row_name]
    return panel


def _num(row: pd.Series, metric: str, sig_rank: str) -> float:
    val = row[(metric, sig_rank)]
    if pd.isna(val):
        return float("nan")
    return float(val)


def engine_to_paper_monthly_pct(spec: FreqSpec, turnover_annualized: float) -> float:
    """GKX monthly turnover % from engine ``Turnover (annualized)`` (= mean τ × periods/year)."""
    if not np.isfinite(turnover_annualized):
        return float("nan")
    per_rebalance = turnover_annualized / spec.periods_per_year
    return per_rebalance / spec.holding_months * 100.0


def _metric_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        out.append(f"{col} Ret (ann.)")
        out.append(f"{col} SR")
    return out


def build_panel_block(
    spec: FreqSpec,
    market: str,
    panel_name: str,
    sheet: str,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_cols = _metric_columns(columns)
    frame_cols = ["Weight scheme", "Row", *metric_cols]
    panel = load_panel(spec, market, sheet)

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

    for row_label, eval_horizon in DH_EVAL_HORIZON_ROWS:
        panel_hk = load_panel(spec, market, sheet, eval_horizon)
        disp_row = {"Weight scheme": panel_name, "Row": row_label}
        raw_ret_row: dict[str, object] = {"Weight scheme": panel_name, "Row": row_label}
        raw_sr_row: dict[str, object] = {"Weight scheme": panel_name, "Row": row_label}
        for col in columns:
            row = panel_hk[col]
            ret = _num(row, RET_METRIC, "DH")
            sr = _num(row, SR_METRIC, "DH")
            ret_key = f"{col} Ret (ann.)"
            sr_key = f"{col} SR"
            raw_ret_row[col] = ret
            raw_sr_row[col] = sr
            t = parse_t_stat(row[(RET_METRIC, "t")])
            disp_row[ret_key] = format_cell(ret, t, value_decimals=2, t_decimals=2)
            disp_row[sr_key] = ""
        display_rows.append(disp_row)
        raw_ret_rows.append(raw_ret_row)
        raw_sr_rows.append(raw_sr_row)

    to_label = "Turnover"
    disp_to: dict[str, object] = {"Weight scheme": panel_name, "Row": to_label}
    raw_to_row: dict[str, object] = {"Weight scheme": panel_name, "Row": to_label}
    for col in columns:
        row = panel[col]
        to_monthly = engine_to_paper_monthly_pct(spec, _num(row, TO_METRIC, "DH"))
        ret_key = f"{col} Ret (ann.)"
        sr_key = f"{col} SR"
        raw_to_row[col] = to_monthly
        disp_to[ret_key] = f"{to_monthly:.0f}%" if np.isfinite(to_monthly) else ""
        disp_to[sr_key] = ""
    display_rows.append(disp_to)
    raw_ret_rows.append(raw_to_row)
    raw_to_rows.append(raw_to_row)

    display = pd.DataFrame(display_rows, columns=frame_cols)
    raw_ret = pd.DataFrame(raw_ret_rows, columns=["Weight scheme", "Row", *columns])
    raw_sr = pd.DataFrame(raw_sr_rows, columns=["Weight scheme", "Row", *columns])
    raw_to = pd.DataFrame(raw_to_rows, columns=["Weight scheme", "Row", *columns])
    return display, raw_ret, raw_sr, raw_to


def build_tables(
    spec: FreqSpec,
    market: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [c[0] for c in spec.cnn_columns] + [c[0] for c in spec.benchmark_columns]
    display_parts: list[pd.DataFrame] = []
    raw_ret_parts: list[pd.DataFrame] = []
    raw_sr_parts: list[pd.DataFrame] = []
    raw_to_parts: list[pd.DataFrame] = []

    for panel_name, sheet, _desc in panels_for_market(market):
        disp, raw_ret, raw_sr, raw_to = build_panel_block(spec, market, panel_name, sheet, columns)
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


def build_notes(spec: FreqSpec, market: str) -> pd.DataFrame:
    rows = [
        {"item": "market", "value": market},
        {"item": "frequency", "value": spec.name},
        {"item": "holding_horizon", "value": spec.label},
    ]
    for panel_name, sheet, desc in panels_for_market(market):
        rows.append({"item": f"panel: {panel_name}", "value": f"source sheet={sheet!r}; {desc}"})
    for line in spec.notes_lines():
        rows.append({"item": "note", "value": line})
    return pd.DataFrame(rows)


def output_path(spec: FreqSpec, market: str) -> Path:
    return HERE / f"{spec.stem}_{market}.xlsx"


def write_excel(
    spec: FreqSpec,
    market: str,
    display: pd.DataFrame,
    raw_returns: pd.DataFrame,
    raw_sharpe: pd.DataFrame,
    raw_turnover: pd.DataFrame,
    notes: pd.DataFrame,
) -> Path:
    out = output_path(spec, market)
    tmp = out.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="notes", index=False)
        display.to_excel(writer, sheet_name="display", index=False)
        raw_returns.to_excel(writer, sheet_name="raw_returns", index=False)
        raw_sharpe.to_excel(writer, sheet_name="raw_sharpe", index=False)
        raw_turnover.to_excel(writer, sheet_name="raw_turnover", index=False)
    tmp.replace(out)
    return out


def run_one(spec: FreqSpec, market: str) -> Path:
    display, raw_ret, raw_sr, raw_to = build_tables(spec, market)
    notes = build_notes(spec, market)
    return write_excel(spec, market, display, raw_ret, raw_sr, raw_to, notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="01 portfolio performance summary (H1–H3)")
    parser.add_argument("--market", choices=(MARKET_US, MARKET_CN), default=MARKET_US)
    parser.add_argument("--freq", choices=tuple(FREQ_SPECS), default="weekly")
    parser.add_argument(
        "--all-freqs",
        action="store_true",
        help="write weekly, monthly, and quarterly tables for the given market",
    )
    args = parser.parse_args()

    freqs = tuple(FREQ_SPECS) if args.all_freqs else (args.freq,)
    for freq in freqs:
        spec = FREQ_SPECS[freq]
        out = run_one(spec, args.market)
        log(f"wrote {out}")


if __name__ == "__main__":
    main()
