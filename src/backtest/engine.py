"""Cross-sectional H1 backtest: quantile groups, three weight schemes, H1Perf metrics.

Signal and cap are as-of formation. Holding return on the same row must already be
the forward period return (loaders attach any lag). No future cap or future signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.markets import MarketSpec
from backtest.newey_west import NW_LAGS_AUTO, format_nw_t, nw_mean_tstat
from config import BACKTEST_N_GROUP, BACKTEST_WEIGHT_SCHEMES

H1_METRICS = (
    "Annualized Return",
    "Annualized Risk",
    "Sharpe Ratio",
    "Annualized Active Return",
    "Annualized Active Risk",
    "Information Ratio",
    "Max Drawdown (Raw)",
    "Max Drawdown (Active)",
    "Turnover (annualized)",
)

_ROUND = {
    "Annualized Return": 4,
    "Annualized Risk": 4,
    "Sharpe Ratio": 3,
    "Annualized Active Return": 4,
    "Annualized Active Risk": 4,
    "Information Ratio": 3,
    "Max Drawdown (Raw)": 4,
    "Max Drawdown (Active)": 4,
    "Turnover (annualized)": 2,
}


def group_label(rank: int) -> str:
    return f"D{int(rank):02d}"


def standardize_by_date(df: pd.DataFrame, date_col: str, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional z-score, winsorize ±3, z-score again (same as SigWeekTest)."""
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    g = out.groupby(date_col, sort=False)[cols]
    z = (out[cols] - g.transform("mean")) / g.transform("std")
    out[cols] = z.clip(lower=-3.0, upper=3.0)
    g = out.groupby(date_col, sort=False)[cols]
    out[cols] = (out[cols] - g.transform("mean")) / g.transform("std")
    return out


def assign_quantile_groups(signal: pd.Series, date: pd.Series, ngroup: int) -> pd.Series:
    if ngroup < 2:
        raise ValueError(f"ngroup must be >= 2, got {ngroup}")
    tmp = pd.DataFrame({"sig": signal.to_numpy(), "date": date.to_numpy()})
    counts = tmp.groupby("date", sort=False)["sig"].size()
    if (counts < ngroup).any():
        n_bad = int((counts < ngroup).sum())
        raise ValueError(f"{n_bad} dates have fewer names than ngroup={ngroup}")
    g = tmp.groupby("date", sort=False)["sig"]
    rank = g.rank(method="first")
    n = g.transform("count")
    grp = np.floor((rank - 1.0) / n * ngroup).astype(np.int32) + 1
    grp = grp.clip(lower=1, upper=ngroup)
    return pd.Series(grp.to_numpy(), index=signal.index, dtype=np.int32)


def _max_dd(returns: np.ndarray) -> float:
    log_r = np.log1p(returns)
    cum = np.exp(np.cumsum(log_r))
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(-np.min(dd))


def _max_dd_by_group(dates: np.ndarray, groups: np.ndarray, values: np.ndarray) -> pd.Series:
    tmp = pd.DataFrame({"date": dates, "g": groups, "r": values})
    tmp = tmp.sort_values(["g", "date"], kind="mergesort")
    out: dict[object, float] = {}
    for g, part in tmp.groupby("g", sort=False):
        out[g] = _max_dd(part["r"].to_numpy(dtype=np.float64))
    return pd.Series(out)


def _weighted_portfolios(
    frame: pd.DataFrame,
    date_col: str,
    group_col: str,
    ret_col: str,
    weight: np.ndarray,
) -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            date_col: frame[date_col].to_numpy(),
            group_col: frame[group_col].to_numpy(),
            "_w": np.asarray(weight, dtype=np.float64),
            "_wr": np.asarray(weight, dtype=np.float64) * frame[ret_col].to_numpy(dtype=np.float64),
        }
    )
    g = tmp.groupby([date_col, group_col], sort=False)
    port = (g["_wr"].sum() / g["_w"].sum()).rename("ret")
    mkt = tmp.groupby(date_col, sort=False)
    avg = (mkt["_wr"].sum() / mkt["_w"].sum()).rename("avg_ret")
    out = port.reset_index().merge(avg.reset_index(), on=date_col, how="inner")
    out["act_ret"] = out["ret"] - out["avg_ret"]
    return out


def _two_way_turnover(
    frame: pd.DataFrame,
    date_col: str,
    id_col: str,
    group_col: str,
    weight: np.ndarray,
) -> pd.Series:
    tmp = pd.DataFrame(
        {
            id_col: frame[id_col].to_numpy(),
            date_col: frame[date_col].to_numpy(),
            group_col: frame[group_col].to_numpy(),
            "_w": np.asarray(weight, dtype=np.float64),
        }
    )
    wsum = tmp.groupby([date_col, group_col], sort=False)["_w"].transform("sum")
    tmp["_wt"] = tmp["_w"] / wsum
    dates = pd.DatetimeIndex(np.sort(pd.unique(pd.to_datetime(tmp[date_col]))))
    prev_of = dict(zip(dates[1:], dates[:-1]))
    cur = tmp[[id_col, date_col, group_col, "_wt"]].copy()
    cur[date_col] = pd.to_datetime(cur[date_col])
    fut = cur.copy()
    fut[date_col] = fut[date_col].map(prev_of)
    fut = fut.dropna(subset=[date_col]).rename(columns={"_wt": "_wt1"})
    both = cur.merge(fut, on=[id_col, date_col, group_col], how="outer")
    both["_wt"] = both["_wt"].fillna(0.0)
    both["_wt1"] = both["_wt1"].fillna(0.0)
    both["dwt"] = (both["_wt"] - both["_wt1"]).abs()
    return both.groupby([date_col, group_col], sort=False)["dwt"].sum()


def _scheme_weights(frame: pd.DataFrame, cap_col: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    if cap_col is None:
        w = np.ones(len(frame), dtype=np.float64)
        return frame, w
    cap = frame[cap_col].to_numpy(dtype=np.float64)
    ok = np.isfinite(cap) & (cap > 0.0)
    kept = frame.iloc[np.flatnonzero(ok)]
    return kept, cap[ok]


def _portfolio_raw_pivot(
    panel: pd.DataFrame,
    *,
    spec: MarketSpec,
    signal_col: str,
    scheme: str,
    ngroup: int = BACKTEST_N_GROUP,
    direct_signal: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Decile portfolio raw/active return pivots and per-group turnover."""
    if scheme not in BACKTEST_WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight scheme: {scheme}")
    date_col = spec.date_col
    id_col = spec.id_col
    ret_col = spec.ret_col
    need = [id_col, date_col, signal_col, ret_col]
    missing = [c for c in need if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing columns {missing}")

    extra: list[str] = []
    for col in (spec.float_cap_col, spec.total_cap_col):
        if col in panel.columns and col not in need and col not in extra:
            extra.append(col)
    work = panel[need + extra].copy()
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[signal_col, ret_col])
    assert len(work) > 0, f"empty panel after dropping NA {signal_col}/{ret_col}"

    if not direct_signal:
        work = standardize_by_date(work, date_col, [signal_col])
        work = work.dropna(subset=[signal_col])
        assert len(work) > 0, f"empty panel after standardizing {signal_col}"
    work = work.copy()
    work["_g"] = assign_quantile_groups(work[signal_col], work[date_col], ngroup)

    cap_col = spec.cap_col(scheme)
    if cap_col is not None and cap_col not in work.columns:
        raise KeyError(f"{spec.name} panel missing {scheme} cap column {cap_col}")
    held, weight = _scheme_weights(work, cap_col)
    assert len(held) > 0, f"empty {scheme} book after requiring positive {cap_col}"

    port = _weighted_portfolios(held, date_col, "_g", ret_col, weight)
    to_g = _two_way_turnover(held, date_col, id_col, "_g", weight)

    raw = port.pivot(index=date_col, columns="_g", values="ret")
    act = port.pivot(index=date_col, columns="_g", values="act_ret")
    raw["DH"] = raw[ngroup] - raw[1]
    act["DH"] = act[ngroup] - act[1]
    return raw, act, to_g


def hl_return_series(
    panel: pd.DataFrame,
    *,
    spec: MarketSpec,
    signal_col: str,
    scheme: str = "equal",
    ngroup: int = BACKTEST_N_GROUP,
    direct_signal: bool = False,
) -> pd.Series:
    """Equal-weight (or scheme) H–L portfolio return at each formation date."""
    raw, _, _ = _portfolio_raw_pivot(
        panel,
        spec=spec,
        signal_col=signal_col,
        scheme=scheme,
        ngroup=ngroup,
        direct_signal=direct_signal,
    )
    out = raw["DH"].copy()
    out.index = pd.to_datetime(out.index)
    out.name = signal_col
    return out.sort_index()


def h1_perf_one(
    panel: pd.DataFrame,
    *,
    spec: MarketSpec,
    signal_col: str,
    scheme: str,
    ngroup: int = BACKTEST_N_GROUP,
    direct_signal: bool = False,
    newey_west_lags: object = NW_LAGS_AUTO,
) -> pd.DataFrame:
    """One signal, one weight scheme: rows=groups (D01..DH, t), columns=H1 metrics."""
    date_col = spec.date_col
    raw, act, to_g = _portfolio_raw_pivot(
        panel,
        spec=spec,
        signal_col=signal_col,
        scheme=scheme,
        ngroup=ngroup,
        direct_signal=direct_signal,
    )

    ppy = spec.periods_per_year
    mean_raw = ppy * raw.mean()
    std_raw = np.sqrt(ppy) * raw.std()
    mean_act = ppy * act.mean()
    std_act = np.sqrt(ppy) * act.std()

    raw_long = raw.reset_index().melt(id_vars=date_col, var_name="_g", value_name="r")
    act_long = act.reset_index().melt(id_vars=date_col, var_name="_g", value_name="r")
    maxdd_raw = _max_dd_by_group(
        raw_long[date_col].to_numpy(), raw_long["_g"].to_numpy(), raw_long["r"].to_numpy()
    )
    maxdd_act = _max_dd_by_group(
        act_long[date_col].to_numpy(), act_long["_g"].to_numpy(), act_long["r"].to_numpy()
    )

    to_mean = to_g.groupby(level="_g").mean() * ppy
    to_mean.loc["DH"] = (to_mean.loc[1] + to_mean.loc[ngroup]) / 2.0

    _, hl_nw_t = nw_mean_tstat(raw["DH"], newey_west_lags=newey_west_lags)

    labels = [group_label(i) for i in range(1, ngroup + 1)] + ["DH"]
    keys = list(range(1, ngroup + 1)) + ["DH"]
    perf = pd.DataFrame(index=labels)
    perf.index.name = "sig_rank"
    perf["Annualized Return"] = mean_raw.reindex(keys).to_numpy()
    perf["Annualized Risk"] = std_raw.reindex(keys).to_numpy()
    perf["Sharpe Ratio"] = perf["Annualized Return"] / perf["Annualized Risk"]
    perf["Annualized Active Return"] = mean_act.reindex(keys).to_numpy()
    perf["Annualized Active Risk"] = std_act.reindex(keys).to_numpy()
    perf["Information Ratio"] = perf["Annualized Active Return"] / perf["Annualized Active Risk"]
    perf["Max Drawdown (Raw)"] = maxdd_raw.reindex(keys).to_numpy()
    perf["Max Drawdown (Active)"] = maxdd_act.reindex(keys).to_numpy()
    to_aligned = to_mean.reindex(keys)
    perf["Turnover (annualized)"] = to_aligned.to_numpy()
    perf = perf.round(_ROUND)
    t_row = pd.Series({m: np.nan for m in H1_METRICS}, name="t", dtype=object)
    t_row["Annualized Return"] = format_nw_t(hl_nw_t)
    perf = pd.concat([perf, t_row.to_frame().T])
    return perf


def format_h1_row(perf: pd.DataFrame, row_name: str) -> pd.DataFrame:
    stacked = perf.stack()
    stacked.index.names = ["sig_rank", "metric"]
    out = stacked.to_frame(row_name).reset_index()
    metric_order = {m: i for i, m in enumerate(H1_METRICS)}
    out["_m"] = out["metric"].map(metric_order)
    out["_g"] = pd.Categorical(
        out["sig_rank"],
        categories=list(perf.index),
        ordered=True,
    )
    out = out.sort_values(["_m", "_g"])
    return out.set_index(["metric", "sig_rank"])[[row_name]].T


def h1_perf_tables(
    panel: pd.DataFrame,
    *,
    spec: MarketSpec,
    signal_cols: list[str],
    ngroup: int = BACKTEST_N_GROUP,
    row_names: list[str] | None = None,
    schemes: tuple[str, ...] = BACKTEST_WEIGHT_SCHEMES,
    direct_signal: bool = False,
    newey_west_lags: object = NW_LAGS_AUTO,
) -> dict[str, pd.DataFrame]:
    """Default output: three H1Perf tables keyed equal / float / total."""
    if row_names is None:
        names = list(signal_cols)
    else:
        if len(row_names) != len(signal_cols):
            raise ValueError("row_names must match signal_cols")
        names = list(row_names)
    tables: dict[str, pd.DataFrame] = {}
    for scheme in schemes:
        rows = [
            format_h1_row(
                h1_perf_one(
                    panel,
                    spec=spec,
                    signal_col=col,
                    scheme=scheme,
                    ngroup=ngroup,
                    direct_signal=direct_signal,
                    newey_west_lags=newey_west_lags,
                ),
                name,
            )
            for col, name in zip(signal_cols, names)
        ]
        tables[scheme] = pd.concat(rows, axis=0)
    return tables
