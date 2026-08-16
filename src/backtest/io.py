"""Load US CNN OOS panels / China signal-return panels, and write H1 Excel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.markets import CN_SPEC, MarketSpec
from config import BACKTEST_WEIGHT_SCHEMES, TEST_START, image_bundle_dir
from models.dataset import model_run_tag


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(path)
    if name.endswith(".feather") or name.endswith(".ft"):
        return pd.read_feather(path)
    if name.endswith(".parquet") or name.endswith(".parquet.gzip") or name.endswith(".gzip"):
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table suffix: {path}")


def ensemble_pred_path(
    models_root: Path,
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    tag = model_run_tag(image_days, horizon, init_from_image_days=init_from_image_days)
    return Path(models_root) / tag / "ensemble_pred.feather"


def load_image_labels(
    images_root: Path,
    image_days: int,
    sample_freq: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    freq_dir = Path(images_root) / image_bundle_dir(image_days, sample_freq)
    paths = sorted(freq_dir.glob(f"{image_days}d_*_labels.feather"))
    if not paths:
        raise FileNotFoundError(f"no label feathers in {freq_dir}")
    ret_cols = [f"Ret_{h}d" for h in horizons]
    cols = ["StockID", "Date", "MarketCap"] + ret_cols
    labels = pd.concat([pd.read_feather(p) for p in paths], ignore_index=True)
    extra = [c for c in ("FloatCap", "TotalCap") if c in labels.columns]
    keep = cols + extra
    labels = labels[keep]
    labels["PERMNO"] = labels["StockID"].astype("int64")
    labels["Date"] = pd.to_datetime(labels["Date"])
    labels = labels.drop(columns=["StockID"])
    if labels.duplicated(["PERMNO", "Date"]).any():
        raise ValueError(f"duplicate PERMNO+Date in image labels under {freq_dir}")
    return labels


def load_us_image_labels(
    images_root: Path,
    image_days: int,
    sample_freq: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    return load_image_labels(images_root, image_days, sample_freq, horizons)


def load_us_predictions(
    models_root: Path,
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> pd.DataFrame:
    pred_path = ensemble_pred_path(
        models_root,
        image_days,
        horizon,
        init_from_image_days=init_from_image_days,
    )
    if not pred_path.is_file():
        raise FileNotFoundError(f"missing OOS predictions: {pred_path}")
    pred = pd.read_feather(pred_path)
    pred["PERMNO"] = pred["PERMNO"].astype("int64")
    pred["Date"] = pd.to_datetime(pred["Date"])
    if pred.duplicated(["PERMNO", "Date"]).any():
        raise ValueError(f"duplicate PERMNO+Date in {pred_path}")
    return pred


def merge_cnn_panel(
    pred: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    horizon: int,
    spec: MarketSpec,
    start: str = TEST_START,
    extra_ret_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Join OOS p_up at formation date t with Ret_{R}d on the same row.

    Ret_{R}d is the forward holding-period return from t+1 through t+R
    (see data.labels.stock_label_panel). No extra signal lag is applied here;
    same-date merge is the T+1 H1 alignment for CNN image labels.
    """
    ret_col = f"Ret_{horizon}d"
    label_cols = ["PERMNO", "Date", ret_col]
    for col in extra_ret_cols:
        if col not in label_cols:
            label_cols.append(col)
    if "FloatCap" in labels.columns and "TotalCap" in labels.columns:
        label_cols.extend(["FloatCap", "TotalCap"])
    else:
        label_cols.append("MarketCap")
    missing = [c for c in label_cols if c not in labels.columns]
    if missing:
        raise KeyError(f"labels missing columns {missing}")
    panel = pred.merge(labels[label_cols], on=["PERMNO", "Date"], how="inner")
    panel = panel[panel["Date"] >= pd.Timestamp(start)]
    assert len(panel) > 0, (
        f"empty {spec.name} panel after merging predictions and labels R{horizon}"
    )
    return panel


def merge_us_panel(
    pred: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    horizon: int,
    start: str = TEST_START,
) -> pd.DataFrame:
    from backtest.markets import us_spec

    return merge_cnn_panel(
        pred,
        labels,
        horizon=horizon,
        spec=us_spec(horizon, horizon),
        start=start,
    )


def load_us_oos_panel(
    *,
    models_root: Path,
    images_root: Path,
    image_days: int,
    horizon: int,
    sample_freq: str,
    init_from_image_days: int | None = None,
    start: str = TEST_START,
) -> pd.DataFrame:
    """OOS p_up at as_of, joined to PIT MarketCap and forward Ret_{h}d from image labels."""
    pred = load_us_predictions(
        models_root,
        image_days,
        horizon,
        init_from_image_days=init_from_image_days,
    )
    labels = load_us_image_labels(images_root, image_days, sample_freq, (horizon,))
    return merge_us_panel(pred, labels, horizon=horizon, start=start)


def cap_col_for_top_n(
    panel: pd.DataFrame,
    spec: MarketSpec,
    *,
    cap_kind: str = "total",
) -> str:
    """PIT cap column for top-N universe (US: MarketCap when no TotalCap)."""
    if cap_kind == "float":
        if spec.float_cap_col not in panel.columns:
            raise KeyError(f"panel missing float cap column {spec.float_cap_col}")
        return spec.float_cap_col
    if cap_kind == "total":
        if spec.total_cap_col in panel.columns:
            return spec.total_cap_col
        if "MarketCap" in panel.columns:
            return "MarketCap"
        raise KeyError(
            f"panel missing cap column for top-N filter "
            f"(need {spec.total_cap_col} or MarketCap)"
        )
    raise ValueError(f"unknown cap_kind={cap_kind!r}, expected 'total' or 'float'")


def filter_top_n_by_cap(
    panel: pd.DataFrame,
    *,
    spec: MarketSpec,
    top_n: int,
    cap_kind: str = "total",
) -> pd.DataFrame:
    """Keep largest top_n names by formation-date cap (PIT; no future cap)."""
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    cap_col = cap_col_for_top_n(panel, spec, cap_kind=cap_kind)
    date_col = spec.date_col
    cap = panel[cap_col].to_numpy(dtype=np.float64)
    ok = np.isfinite(cap) & (cap > 0.0)
    tmp = panel.iloc[np.flatnonzero(ok)]
    rank = tmp.groupby(date_col, sort=False)[cap_col].rank(method="first", ascending=False)
    out = tmp.loc[rank <= top_n]
    assert len(out) > 0, f"empty panel after top-{top_n} filter on {cap_col}"
    return out


def apply_universe(
    df: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
) -> pd.DataFrame:
    u = universe[[id_col, date_col]].copy()
    u[date_col] = pd.to_datetime(u[date_col])
    out = df.merge(u, on=[id_col, date_col], how="inner")
    assert len(out) > 0, "empty panel after universe filter"
    return out


def attach_holding_return(
    signals: pd.DataFrame,
    period_returns: pd.DataFrame,
    *,
    spec: MarketSpec,
    lag: int,
) -> pd.DataFrame:
    """Keep formation-date signal and caps; attach period return at date rank t+lag.

    lag=0: return already on the signal date (US forward labels).
    lag=1: next rebalance-period return (China weekly test).
    """
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    id_col = spec.id_col
    date_col = spec.date_col
    ret_col = spec.ret_col
    sig = signals.copy()
    sig[date_col] = pd.to_datetime(sig[date_col])
    rets = period_returns[[id_col, date_col, ret_col]].copy()
    rets[date_col] = pd.to_datetime(rets[date_col])
    if lag == 0:
        out = sig.merge(rets, on=[id_col, date_col], how="inner")
        assert len(out) > 0, "empty panel after aligning same-date returns"
        return out

    dates = pd.DatetimeIndex(pd.unique(rets[date_col])).sort_values()
    rank = pd.Series(np.arange(len(dates), dtype=np.int32), index=dates)
    rets = rets.copy()
    rets["_rank"] = rets[date_col].map(rank)
    sig["_rank"] = sig[date_col].map(rank)
    unmatched = int(sig["_rank"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} signal rows have {date_col} values not on the return calendar"
        )
    rets["_rank"] = rets["_rank"] - np.int32(lag)
    rets = rets.drop(columns=[date_col])
    out = sig.merge(rets, on=[id_col, "_rank"], how="inner")
    out = out.drop(columns=["_rank"])
    assert len(out) > 0, f"empty panel after attaching return lag={lag}"
    return out


def load_cn_panel(
    signals: pd.DataFrame,
    period_returns: pd.DataFrame,
    *,
    spec: MarketSpec = CN_SPEC,
    lag: int = 1,
    universe: pd.DataFrame | None = None,
    start: str | None = None,
) -> pd.DataFrame:
    """China factor panel: formation-date caps on `signals`, next-period `ret` from returns."""
    sig = signals.copy()
    sig[spec.date_col] = pd.to_datetime(sig[spec.date_col])
    sig[spec.id_col] = sig[spec.id_col].astype(str)
    for col in (spec.float_cap_col, spec.total_cap_col):
        if col not in sig.columns:
            raise KeyError(
                f"China signals missing {col}; caps must be as-of the signal date"
            )
    rets = period_returns.copy()
    rets[spec.date_col] = pd.to_datetime(rets[spec.date_col])
    rets[spec.id_col] = rets[spec.id_col].astype(str)
    if spec.ret_col not in rets.columns:
        raise KeyError(f"China returns missing {spec.ret_col}")
    if universe is not None:
        univ = universe.copy()
        univ[spec.date_col] = pd.to_datetime(univ[spec.date_col])
        univ[spec.id_col] = univ[spec.id_col].astype(str)
        sig = apply_universe(sig, univ, id_col=spec.id_col, date_col=spec.date_col)
    panel = attach_holding_return(sig, rets, spec=spec, lag=lag)
    if start is not None:
        panel = panel[panel[spec.date_col] >= pd.Timestamp(start)]
        assert len(panel) > 0, f"empty China panel after start={start}"
    return panel


def write_h1_excel(tables: dict[str, pd.DataFrame], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for scheme in BACKTEST_WEIGHT_SCHEMES:
            tables[scheme].to_excel(writer, sheet_name=scheme)
    tmp.replace(path)
    return path
