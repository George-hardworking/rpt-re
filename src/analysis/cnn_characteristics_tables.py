"""Table V–VIII: CNN vs stock characteristics (correlations and panel logit)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.table_format import (
    build_display_matrix,
    format_value_only,
    write_display_raw_excel,
)
from backtest.io import load_us_predictions
from backtest.newey_west import nw_mean_tstat
from config import (
    TABLE_V_CHAR_COLS,
    TABLE_V_CHAR_DISPLAY,
    TABLE_V_CNN_CONFIGS,
    TABLE_VI_CNN_CONFIGS,
    TABLE_VI_CHAR_COLS,
    TABLE_VII_IMAGE_DAYS,
    TABLE_VIII_HORIZONS,
    TABLE_VIII_IMAGE_DAYS,
    benchmark_tables_output_dir,
    characteristics_month_end_path,
    market_sample_config,
)
from data.images import (
    build_window_arrays,
    chain_adjust_from_first_day,
    moving_average_window_scale,
    observed_dates,
    window_stock_days,
)
from data.labels import stock_label_panel
from data.parquet_io import load_calendar, permno_list, read_stock
from data.trend_signals import (
    MA_COLS,
    compute_hzz_trend_scores,
    liquidity_partition_complete,
    read_stock_trend_signals,
)

MIN_CROSS_SECTION = 30
LOGIT_MAX_ITER = 100
LOGIT_TOL = 1e-8
TRAIN_MIN_OBS = 12
IMAGE_REGressor_PREFIXES = ("open", "high", "low", "close", "ma", "vol")
IMAGE_LAGS = (1, 2, 3, 4, 5)
VIII_EXCLUDE_REGRESSORS = frozenset({"close_lag5"})


def _cnn_col(image_days: int, horizon: int) -> str:
    return f"p_up_I{image_days}_R{horizon}"


def _config_row_label(image_days: int, horizon: int) -> str:
    return f"I{image_days}/R{horizon}"


def month_end_dates(calendar: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    cal = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if len(cal) == 0:
        raise ValueError(f"empty calendar between {start} and {end}")
    by_month = pd.Series(cal, index=cal).groupby(cal.to_period("M")).max()
    return pd.DatetimeIndex(by_month.sort_index().values)


def _size_from_ohlc(stock_df: pd.DataFrame, as_of: pd.Timestamp) -> float:
    row = stock_df.loc[stock_df["DlyCalDt"] == as_of]
    if row.empty:
        return float("nan")
    if "DlyTotalCap" in stock_df.columns:
        return float(row["DlyTotalCap"].iloc[0])
    return float(row["DlyCap"].iloc[0])


def _load_predictions(
    models_root: Path,
    configs: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], pd.DataFrame]:
    out: dict[tuple[int, int], pd.DataFrame] = {}
    for image_days, horizon in configs:
        pred = load_us_predictions(models_root, image_days, horizon)
        col = _cnn_col(image_days, horizon)
        out[(image_days, horizon)] = pred.rename(columns={"p_up": col})[
            ["PERMNO", "Date", col]
        ]
    return out


def _attach_predictions(
    panel: pd.DataFrame,
    pred_map: dict[tuple[int, int], pd.DataFrame],
) -> pd.DataFrame:
    out = panel.sort_values(["PERMNO", "Date"]).copy()
    for (_, _), pred in pred_map.items():
        col = [c for c in pred.columns if c.startswith("p_up_")][0]
        merged = pd.merge_asof(
            out,
            pred.sort_values(["PERMNO", "Date"]),
            by="PERMNO",
            left_on="Date",
            right_on="Date",
            direction="backward",
        )
        out[col] = merged[col]
    return out


def build_characteristics_month_end_panel(
    *,
    market: str,
    ohlc_path: Path,
    signals_root: Path,
    models_root: Path,
    out_path: Path | None = None,
) -> pd.DataFrame:
    sample = market_sample_config(market)
    calendar = load_calendar(ohlc_path)
    month_ends = month_end_dates(calendar, sample.test_start, sample.sample_end)

    pred_map = _load_predictions(models_root, TABLE_V_CNN_CONFIGS)
    permnos = permno_list(ohlc_path)
    rows: list[dict] = []

    for permno in permnos:
        permno = int(permno)
        if not liquidity_partition_complete(signals_root, permno):
            continue
        stock_df = read_stock(ohlc_path, permno)
        stock_df = stock_df.copy()
        stock_df["DlyCalDt"] = pd.to_datetime(stock_df["DlyCalDt"])
        signals = read_stock_trend_signals(signals_root, permno).set_index("DlyCalDt")
        labels = stock_label_panel(stock_df)

        for me in month_ends:
            if me not in signals.index or me not in labels.index:
                continue
            sig = signals.loc[me]
            lab = labels.loc[me]
            row: dict = {
                "PERMNO": permno,
                "Date": me,
                "Ret_5d": float(lab["Ret_5d"]),
                "Ret_20d": float(lab["Ret_20d"]),
                "Ret_60d": float(lab["Ret_60d"]),
                "Size": _size_from_ohlc(stock_df, me),
            }
            for col in TABLE_VI_CHAR_COLS:
                if col == "TREND_HZZ":
                    continue
                if col == "Size":
                    continue
                row[col] = float(sig[col])
            for ma in MA_COLS:
                row[ma] = float(sig[ma])
            rows.append(row)

    if not rows:
        raise ValueError(f"empty month-end panel market={market}")

    panel = pd.DataFrame(rows)
    hzz = compute_hzz_trend_scores(panel, ret_col="Ret_5d")
    panel["TREND_HZZ"] = hzz.reindex(panel.index).to_numpy(dtype=np.float32)
    panel = _attach_predictions(panel, pred_map)

    path = out_path if out_path is not None else characteristics_month_end_path(market)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return panel


def cross_section_ranks(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for d, grp in df.groupby("Date", sort=False):
        idx = grp.index
        for col in cols:
            out.loc[idx, col] = grp[col].rank(method="average").to_numpy(dtype=np.float64)
    return out


def cross_section_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < MIN_CROSS_SECTION:
        return float("nan")
    return float(x[mask].rank().corr(y[mask].rank()))


def fit_logit_irls(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Binary logit with intercept; x shape (n, k) without intercept column."""
    n, k = x.shape
    design = np.column_stack([np.ones(n, dtype=np.float64), x])
    beta = np.zeros(k + 1, dtype=np.float64)
    y = y.astype(np.float64)

    for _ in range(LOGIT_MAX_ITER):
        eta = design @ beta
        eta = np.clip(eta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-8, None)
        z = eta + (y - p) / w
        wx = design * w[:, None]
        lhs = design.T @ wx
        rhs = design.T @ (w * z)
        beta_new = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(beta_new - beta)) < LOGIT_TOL:
            beta = beta_new
            break
        beta = beta_new
    return beta


def log_likelihood(y: np.ndarray, x: np.ndarray, beta: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y), dtype=np.float64), x])
    eta = np.clip(design @ beta, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def mcfadden_r2(y: np.ndarray, x: np.ndarray, beta: np.ndarray) -> float:
    ll_m = log_likelihood(y, x, beta)
    p0 = float(np.clip(y.mean(), 1e-12, 1.0 - 1e-12))
    ll0 = float(np.sum(y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0)))
    if ll0 == 0.0:
        return float("nan")
    return float(1.0 - ll_m / ll0) * 100.0


def fama_macbeth_logit(
    panel: pd.DataFrame,
    *,
    y_col: str,
    x_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Monthly cross-section logit; return coef time series, mean, NW t."""
    coef_names = ("const",) + x_cols
    series: list[pd.Series] = []

    for _, grp in panel.groupby("Date", sort=True):
        sub = grp[[y_col, *x_cols]].dropna()
        if len(sub) < MIN_CROSS_SECTION:
            continue
        y = sub[y_col].to_numpy(dtype=np.float64)
        x = sub[list(x_cols)].to_numpy(dtype=np.float64)
        beta = fit_logit_irls(y, x)
        series.append(pd.Series(beta, index=coef_names, name=grp["Date"].iloc[0]))

    if not series:
        raise ValueError(f"no valid cross-sections for y={y_col} x={x_cols}")

    coef_ts = pd.DataFrame(series)
    means = coef_ts.mean(axis=0)
    t_stats = pd.Series(index=coef_names, dtype=np.float64)
    for name in coef_names:
        _, t = nw_mean_tstat(coef_ts[name])
        t_stats[name] = t
    return coef_ts, means, t_stats


def pooled_logit_mcfadden(
    panel: pd.DataFrame,
    *,
    y_col: str,
    x_cols: tuple[str, ...],
) -> float:
    sub = panel[[y_col, *x_cols]].dropna()
    y = sub[y_col].to_numpy(dtype=np.float64)
    x = sub[list(x_cols)].to_numpy(dtype=np.float64)
    beta = fit_logit_irls(y, x)
    return mcfadden_r2(y, x, beta)


def _stock_positive_benchmark(
    train_panel: pd.DataFrame,
    ret_col: str,
) -> tuple[pd.Series, float]:
    y = (train_panel[ret_col] > 0.0).astype(float)
    counts = train_panel.groupby("PERMNO").size()
    by_stock = train_panel.assign(_y=y).groupby("PERMNO")["_y"].mean()
    overall = float(y.mean())
    bench = by_stock.copy()
    for permno, cnt in counts.items():
        if cnt < TRAIN_MIN_OBS:
            bench.loc[permno] = overall
    return bench, overall


def oos_mcfadden_r2(
    test_panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    *,
    ret_col: str,
    x_cols: tuple[str, ...],
) -> float:
    sub = test_panel[[ret_col, "PERMNO", *x_cols]].dropna()
    y = (sub[ret_col] > 0.0).astype(float).to_numpy()
    x = sub[list(x_cols)].to_numpy(dtype=np.float64)
    beta = fit_logit_irls(y, x)
    ll_m = log_likelihood(y, x, beta)

    bench, _ = _stock_positive_benchmark(train_panel, ret_col)
    p_b = sub["PERMNO"].map(bench).to_numpy(dtype=np.float64)
    p_b = np.clip(p_b, 1e-12, 1.0 - 1e-12)
    ll_b = float(np.sum(y * np.log(p_b) + (1.0 - y) * np.log(1.0 - p_b)))
    if ll_b == 0.0:
        return float("nan")
    return float(1.0 - ll_m / ll_b) * 100.0


def table_v_correlation(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    char_cols = TABLE_V_CHAR_COLS
    rank_cols = tuple(_cnn_col(i, r) for i, r in TABLE_V_CNN_CONFIGS) + char_cols
    ranked = cross_section_ranks(panel, rank_cols)

    mean_vals = pd.DataFrame(
        index=[_config_row_label(i, r) for i, r in TABLE_V_CNN_CONFIGS],
        columns=[TABLE_V_CHAR_DISPLAY[c] for c in char_cols],
        dtype=np.float64,
    )
    t_stats = mean_vals.copy()

    for image_days, horizon in TABLE_V_CNN_CONFIGS:
        row = _config_row_label(image_days, horizon)
        pcol = _cnn_col(image_days, horizon)
        for char in char_cols:
            monthly: list[float] = []
            for _, grp in ranked.groupby("Date", sort=True):
                rho = cross_section_spearman(grp[pcol], grp[char])
                if np.isfinite(rho):
                    monthly.append(rho)
            if not monthly:
                mean_vals.at[row, TABLE_V_CHAR_DISPLAY[char]] = float("nan")
                t_stats.at[row, TABLE_V_CHAR_DISPLAY[char]] = float("nan")
                continue
            arr = np.asarray(monthly, dtype=np.float64)
            mean, t = nw_mean_tstat(arr)
            mean_vals.at[row, TABLE_V_CHAR_DISPLAY[char]] = mean
            t_stats.at[row, TABLE_V_CHAR_DISPLAY[char]] = t

    display = build_display_matrix(mean_vals, t_stats)
    return display, mean_vals, t_stats


def _prepare_vi_panel(
    panel: pd.DataFrame,
    image_days: int,
    horizon: int,
) -> pd.DataFrame:
    pcol = _cnn_col(image_days, horizon)
    sub = panel.dropna(subset=[pcol, *TABLE_VI_CHAR_COLS]).copy()
    sub["y"] = (sub[pcol] > 0.5).astype(np.float64)
    ranked = cross_section_ranks(sub, TABLE_VI_CHAR_COLS)
    ranked["y"] = sub["y"]
    return ranked


def table_vi_forecast_logit(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    col_labels = [f"{image_days}D{horizon}P" for image_days, horizon in TABLE_VI_CNN_CONFIGS]
    row_labels = [TABLE_V_CHAR_DISPLAY[c] for c in TABLE_VI_CHAR_COLS] + ["McFadden R²"]
    mean_vals = pd.DataFrame(index=row_labels, columns=col_labels, dtype=np.float64)
    t_stats = mean_vals.copy()

    for (image_days, horizon), col_label in zip(TABLE_VI_CNN_CONFIGS, col_labels):
        vi_panel = _prepare_vi_panel(panel, image_days, horizon)
        _, means, ts = fama_macbeth_logit(
            vi_panel,
            y_col="y",
            x_cols=TABLE_VI_CHAR_COLS,
        )
        for char in TABLE_VI_CHAR_COLS:
            mean_vals.at[TABLE_V_CHAR_DISPLAY[char], col_label] = means[char]
            t_stats.at[TABLE_V_CHAR_DISPLAY[char], col_label] = ts[char]
        r2 = pooled_logit_mcfadden(vi_panel, y_col="y", x_cols=TABLE_VI_CHAR_COLS)
        mean_vals.at["McFadden R²", col_label] = r2
        t_stats.at["McFadden R²", col_label] = float("nan")

    display = build_display_matrix(
        mean_vals.drop(index=["McFadden R²"]),
        t_stats.drop(index=["McFadden R²"]),
    )
    for col in col_labels:
        display.loc["McFadden R²", col] = format_value_only(mean_vals.at["McFadden R²", col])
    raw_vals = mean_vals.copy()
    raw_t = t_stats.copy()
    return display, raw_vals, raw_t


def _vii_column_labels() -> list[str]:
    labels: list[str] = []
    for image_days in TABLE_VII_IMAGE_DAYS:
        for spec in ("CNN", "Chars", "Joint"):
            labels.append(f"I{image_days}_R5_{spec}")
    return labels


def table_vii_return_logit(
    panel: pd.DataFrame,
    train_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    col_labels = _vii_column_labels()
    rows = ["CNN"] + [TABLE_V_CHAR_DISPLAY[c] for c in TABLE_VI_CHAR_COLS] + ["OOS McFadden R²"]
    mean_vals = pd.DataFrame(index=rows, columns=col_labels, dtype=np.float64)
    t_stats = mean_vals.copy()

    for image_days in TABLE_VII_IMAGE_DAYS:
        pcol = _cnn_col(image_days, 5)
        specs: list[tuple[str, tuple[str, ...]]] = [
            (f"I{image_days}_R5_CNN", (pcol,)),
            (f"I{image_days}_R5_Chars", TABLE_VI_CHAR_COLS),
            (f"I{image_days}_R5_Joint", (pcol,) + TABLE_VI_CHAR_COLS),
        ]
        for col_label, x_cols in specs:
            sub = panel.dropna(subset=["Ret_5d", *x_cols]).copy()
            sub["y"] = (sub["Ret_5d"] > 0.0).astype(np.float64)
            ranked = cross_section_ranks(sub, x_cols)
            ranked["y"] = sub["y"]
            _, means, ts = fama_macbeth_logit(ranked, y_col="y", x_cols=x_cols)

            if "CNN" in x_cols and pcol in x_cols:
                if pcol in means.index:
                    mean_vals.at["CNN", col_label] = means[pcol]
                    t_stats.at["CNN", col_label] = ts[pcol]
            elif x_cols == (pcol,):
                mean_vals.at["CNN", col_label] = means[pcol]
                t_stats.at["CNN", col_label] = ts[pcol]

            for char in TABLE_VI_CHAR_COLS:
                if char in x_cols:
                    mean_vals.at[TABLE_V_CHAR_DISPLAY[char], col_label] = means[char]
                    t_stats.at[TABLE_V_CHAR_DISPLAY[char], col_label] = ts[char]

            oos = oos_mcfadden_r2(sub, train_panel, ret_col="Ret_5d", x_cols=x_cols)
            mean_vals.at["OOS McFadden R²", col_label] = oos
            t_stats.at["OOS McFadden R²", col_label] = float("nan")

    display = build_display_matrix(
        mean_vals.drop(index=["OOS McFadden R²"]),
        t_stats.drop(index=["OOS McFadden R²"]),
    )
    for col in col_labels:
        display.loc["OOS McFadden R²", col] = format_value_only(
            mean_vals.at["OOS McFadden R²", col]
        )
    return display, mean_vals, t_stats


def _image_minmax_scale(values: np.ndarray, pmin: float, prange: float) -> np.ndarray:
    if prange <= 0.0 or not np.isfinite(prange):
        return np.full_like(values, np.nan, dtype=np.float64)
    return (values - pmin) / prange


def image_scaled_lags(
    stock_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    window_days: int = TABLE_VIII_IMAGE_DAYS,
) -> dict[str, float] | None:
    dates = observed_dates(stock_df)
    window = window_stock_days(dates, as_of, window_days)
    if window is None or len(window) != window_days:
        return None
    if window[-1] != as_of:
        return None

    series = build_window_arrays(stock_df, window)
    if np.any(~np.isfinite(series.close)):
        return None

    chained = chain_adjust_from_first_day(series)
    if chained is None:
        return None
    adj_open, adj_high, adj_low, adj_close = chained
    n_hist = 0
    ma = moving_average_window_scale(adj_close, n_hist, window_days)

    price_stack = np.concatenate(
        [adj_open, adj_high, adj_low, adj_close, ma[np.isfinite(ma)]]
    )
    finite_prices = price_stack[np.isfinite(price_stack)]
    if len(finite_prices) == 0:
        return None
    pmin = float(finite_prices.min())
    pmax = float(finite_prices.max())
    prange = pmax - pmin

    vol = series.volume.astype(np.float64)
    vol_max = float(np.nanmax(vol)) if np.any(np.isfinite(vol)) else float("nan")
    if not np.isfinite(vol_max) or vol_max <= 0.0:
        vol_scaled = np.full(window_days, np.nan)
    else:
        vol_scaled = vol / vol_max

    o = _image_minmax_scale(adj_open, pmin, prange)
    h = _image_minmax_scale(adj_high, pmin, prange)
    l = _image_minmax_scale(adj_low, pmin, prange)
    c = _image_minmax_scale(adj_close, pmin, prange)
    m = _image_minmax_scale(ma, pmin, prange)

    out: dict[str, float] = {}
    for lag in IMAGE_LAGS:
        idx = -lag
        out[f"open_lag{lag}"] = float(o[idx])
        out[f"high_lag{lag}"] = float(h[idx])
        out[f"low_lag{lag}"] = float(l[idx])
        out[f"close_lag{lag}"] = float(c[idx])
        out[f"ma_lag{lag}"] = float(m[idx])
        out[f"vol_lag{lag}"] = float(vol_scaled[idx])
    return out


def _viii_regressors() -> tuple[str, ...]:
    cols: list[str] = []
    for prefix in IMAGE_REGressor_PREFIXES:
        for lag in IMAGE_LAGS:
            name = f"{prefix}_lag{lag}"
            if name not in VIII_EXCLUDE_REGRESSORS:
                cols.append(name)
    return tuple(cols)


def _attach_image_features(
    panel: pd.DataFrame,
    ohlc_path: Path,
) -> pd.DataFrame:
    reg_cols = _viii_regressors()
    cache: dict[int, pd.DataFrame] = {}
    feat_rows: list[dict] = []

    for permno in panel["PERMNO"].unique():
        permno = int(permno)
        if permno not in cache:
            df = read_stock(ohlc_path, permno)
            df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
            cache[permno] = df
        stock_df = cache[permno]
        for me in panel.loc[panel["PERMNO"] == permno, "Date"]:
            lags = image_scaled_lags(stock_df, pd.Timestamp(me))
            if lags is None:
                continue
            rec = {"PERMNO": permno, "Date": pd.Timestamp(me)}
            rec.update(lags)
            feat_rows.append(rec)

    if not feat_rows:
        raise ValueError("no image-scaled features built for Table VIII")

    feats = pd.DataFrame(feat_rows)
    merged = panel.merge(feats, on=["PERMNO", "Date"], how="inner")
    missing = [c for c in reg_cols if c not in merged.columns]
    if missing:
        raise KeyError(f"Table VIII missing regressors: {missing}")
    return merged


def _viii_column_specs() -> list[tuple[str, int, tuple[str, ...]]]:
    """Paper Table VIII: cols 1–3 CNN dep by horizon; 4–12 return specs in groups of 3."""
    specs: list[tuple[str, int, tuple[str, ...]]] = []
    reg_cols = _viii_regressors()
    for horizon in TABLE_VIII_HORIZONS:
        specs.append(("y_cnn", horizon, reg_cols))
    for horizon in TABLE_VIII_HORIZONS:
        pcol = _cnn_col(TABLE_VIII_IMAGE_DAYS, horizon)
        specs.append(("y_ret", horizon, (pcol,)))
        specs.append(("y_ret", horizon, reg_cols))
        specs.append(("y_ret", horizon, (pcol,) + reg_cols))
    return specs


def table_viii_image_logit(
    panel: pd.DataFrame,
    train_panel: pd.DataFrame,
    ohlc_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg_cols = _viii_regressors()
    img_panel = _attach_image_features(panel, ohlc_path)
    col_specs = _viii_column_specs()
    col_labels = [f"({i})" for i in range(1, len(col_specs) + 1)]

    row_names = ["CNN"] + list(reg_cols) + ["McFadden R²", "OOS McFadden R²"]
    mean_vals = pd.DataFrame(index=row_names, columns=col_labels, dtype=np.float64)
    t_stats = mean_vals.copy()

    for col_label, (y_kind, horizon, x_cols) in zip(col_labels, col_specs):
        pcol = _cnn_col(TABLE_VIII_IMAGE_DAYS, horizon)
        ret_col = f"Ret_{horizon}d"
        sub = img_panel.dropna(subset=[pcol, ret_col, *x_cols]).copy()
        if y_kind == "y_cnn":
            sub["y"] = (sub[pcol] > 0.5).astype(np.float64)
        else:
            sub["y"] = (sub[ret_col] > 0.0).astype(np.float64)
        ranked = cross_section_ranks(sub, x_cols)
        ranked["y"] = sub["y"]
        _, means, ts = fama_macbeth_logit(ranked, y_col="y", x_cols=x_cols)

        if pcol in x_cols and pcol in means.index:
            mean_vals.at["CNN", col_label] = means[pcol]
            t_stats.at["CNN", col_label] = ts[pcol]
        for reg in reg_cols:
            if reg in x_cols and reg in means.index:
                mean_vals.at[reg, col_label] = means[reg]
                t_stats.at[reg, col_label] = ts[reg]

        if y_kind == "y_cnn":
            mean_vals.at["McFadden R²", col_label] = pooled_logit_mcfadden(
                ranked, y_col="y", x_cols=x_cols
            )
            mean_vals.at["OOS McFadden R²", col_label] = float("nan")
        else:
            mean_vals.at["McFadden R²", col_label] = float("nan")
            mean_vals.at["OOS McFadden R²", col_label] = oos_mcfadden_r2(
                sub, train_panel, ret_col=ret_col, x_cols=x_cols
            )

    coef_rows = [r for r in row_names if r not in ("McFadden R²", "OOS McFadden R²")]
    display = build_display_matrix(mean_vals.loc[coef_rows], t_stats.loc[coef_rows])
    for metric in ("McFadden R²", "OOS McFadden R²"):
        for col in col_labels:
            display.loc[metric, col] = format_value_only(mean_vals.at[metric, col])
    return display, mean_vals, t_stats


def _train_month_end_panel(panel: pd.DataFrame, market: str) -> pd.DataFrame:
    train_end = market_sample_config(market).train_end
    return panel[panel["Date"] <= pd.Timestamp(train_end)].copy()


def write_run_meta(out_dir: Path, market: str) -> Path:
    sample = market_sample_config(market)
    meta = {
        "market": market,
        "test_start": sample.test_start,
        "test_end": sample.sample_end,
        "train_end": sample.train_end,
        "omitted_characteristics": ["Bid-Ask", "Price Delay"],
        "beta_window": 252,
        "vol_window": 21,
        "liquidity_window": 21,
        "table_v_alignment": "month-end; CNN p_up merge_asof backward by PERMNO",
        "display_format": "value + stars + (t)",
    }
    path = out_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def run_all_tables(
    *,
    market: str,
    ohlc_path: Path,
    signals_root: Path,
    models_root: Path,
    fresh: bool,
) -> list[Path]:
    out_dir = benchmark_tables_output_dir(market)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = characteristics_month_end_path(market)

    if panel_path.is_file() and not fresh:
        panel = pd.read_parquet(panel_path)
        panel["Date"] = pd.to_datetime(panel["Date"])
    else:
        if fresh and panel_path.is_file():
            panel_path.unlink()
        panel = build_characteristics_month_end_panel(
            market=market,
            ohlc_path=ohlc_path,
            signals_root=signals_root,
            models_root=models_root,
            out_path=panel_path,
        )

    sample = market_sample_config(market)
    test_panel = panel[panel["Date"] >= pd.Timestamp(sample.test_start)].copy()
    train_panel = _train_month_end_panel(panel, market)

    written: list[Path] = []

    paths = {
        "table_v_correlation.xlsx": table_v_correlation,
        "table_vi_forecast_logit.xlsx": lambda p: table_vi_forecast_logit(p),
        "table_vii_return_logit.xlsx": lambda p: table_vii_return_logit(p, train_panel),
        "table_viii_image_logit.xlsx": lambda p: table_viii_image_logit(p, train_panel, ohlc_path),
    }

    for fname, fn in paths.items():
        out_path = out_dir / fname
        if out_path.is_file() and not fresh:
            written.append(out_path)
            continue
        if fresh and out_path.is_file():
            out_path.unlink()
        display, raw_vals, raw_t = fn(test_panel)
        write_display_raw_excel(
            out_path,
            display=display,
            raw_values=raw_vals,
            raw_t=raw_t,
        )
        written.append(out_path)

    written.append(write_run_meta(out_dir, market))
    return written
