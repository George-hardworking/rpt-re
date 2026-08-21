"""STW Figure-8 pipeline: universe panels, cap columns, and CNN H-L Sharpe lookup."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from backtest.eval_horizon import backtest_output_stem
from backtest.io import load_image_labels
from backtest.markets import MarketSpec, cn_cnn_spec, us_spec
from config import (
    BACKTEST_CNN_ROOT,
    BACKTEST_STW_ROOT,
    BACKTEST_WEIGHT_SCHEMES,
    HORIZON_BACKTEST_DIR,
    MARKET_CN,
    MARKET_US,
    STW_FIGURE8_IMAGE_DAYS,
    market_processed_dir,
    market_sample_config,
    sample_freq_for_horizon,
    stw_figure_dir,
    stw_processed_horizon_dir,
)
from models.dataset import model_run_tag


def stw_data_root(market: str, horizon: int, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override) / HORIZON_BACKTEST_DIR[horizon]
    return stw_processed_horizon_dir(market, horizon)


def stw_figures_root(market: str, horizon: int, override: Path | None = None) -> Path:
    root = Path(override) if override is not None else BACKTEST_STW_ROOT
    return root / market / HORIZON_BACKTEST_DIR[horizon] / "figures"


def stw_market_figures_dir(market: str, figure_root: Path | None = None) -> Path:
    root = Path(figure_root) if figure_root is not None else BACKTEST_STW_ROOT
    return root / market / "figures"


STW_CHECKPOINT_SCAN = ".checkpoint_scan_batches"
STW_FRAGMENTS_SUBDIR = "_tmp_signal_spill/fragments"
STW_UNIVERSE_PANEL = "universe_panel.parquet"


def stw_fragments_dir(data_root: Path) -> Path:
    return Path(data_root) / STW_FRAGMENTS_SUBDIR


def stw_scan_checkpoint_path(data_root: Path) -> Path:
    return Path(data_root) / STW_CHECKPOINT_SCAN


def stw_fragment_path(data_root: Path, batch_id: int, chunk_id: int) -> Path:
    return stw_fragments_dir(data_root) / f"batch_{batch_id:05d}_chunk_{chunk_id:04d}.parquet"


def load_scan_checkpoint(data_root: Path) -> set[int]:
    path = stw_scan_checkpoint_path(data_root)
    if not path.is_file():
        return set()
    done: set[int] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            done.add(int(line))
    return done


def mark_scan_batch_done(data_root: Path, batch_id: int) -> None:
    path = stw_scan_checkpoint_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(f"{batch_id}\n")


def batch_fragments_complete(data_root: Path, batch_id: int, pending_chunk_ids: list[int]) -> bool:
    return all(
        stw_fragment_path(data_root, batch_id, chunk_id).is_file() for chunk_id in pending_chunk_ids
    )


def clear_horizon_scan_state(data_root: Path) -> None:
    spill = Path(data_root) / "_tmp_signal_spill"
    if spill.exists():
        shutil.rmtree(spill)
    ckpt = stw_scan_checkpoint_path(data_root)
    if ckpt.exists():
        ckpt.unlink()


def figure8_complete(market: str, figure_root: Path | None = None) -> bool:
    fig_dir = stw_market_figures_dir(market, figure_root)
    return (fig_dir / "figure8_stw_sharpe_distribution.png").is_file()


def stw_market_spec(market: str, horizon: int) -> MarketSpec:
    if market == MARKET_US:
        return us_spec(STW_FIGURE8_IMAGE_DAYS, horizon)
    if market == MARKET_CN:
        return cn_cnn_spec(horizon)
    raise ValueError(f"unsupported market={market}")


def stw_ohlc_root(market: str, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return market_processed_dir(market) / "ohlc_daily"


def stw_images_root(market: str, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return market_processed_dir(market) / "images"


def stw_universe_from_image_labels(
    market: str,
    horizon: int,
    *,
    images_root: Path,
    start: str | None = None,
    permno_limit: int | None = None,
) -> pd.DataFrame:
    """CNN-aligned rebalance panel: I20 labels at the horizon's sample frequency."""
    sample_freq = sample_freq_for_horizon(horizon)
    labels = load_image_labels(
        images_root,
        STW_FIGURE8_IMAGE_DAYS,
        sample_freq,
        (horizon,),
    )
    spec = stw_market_spec(market, horizon)
    ret_col = spec.ret_col
    cols = ["PERMNO", "Date", ret_col]
    for cap_col in (spec.float_cap_col, spec.total_cap_col):
        if cap_col not in labels.columns:
            raise KeyError(
                f"image labels missing cap column {cap_col} for market={market} "
                f"bundle={STW_FIGURE8_IMAGE_DAYS}d_{sample_freq}; "
                f"run 02_generate_images for I{STW_FIGURE8_IMAGE_DAYS} at {sample_freq} frequency"
            )
        if cap_col not in cols:
            cols.append(cap_col)

    universe = labels[cols].copy()
    universe["Date"] = pd.to_datetime(universe["Date"])
    universe["PERMNO"] = universe["PERMNO"].astype("int64")
    if start is None:
        start = market_sample_config(market).test_start
    universe = universe[universe["Date"] >= pd.Timestamp(start)].copy()
    if permno_limit is not None:
        keep = sorted(universe["PERMNO"].unique())[:permno_limit]
        universe = universe[universe["PERMNO"].isin(keep)].copy()
    if universe.empty:
        raise ValueError(
            f"empty STW universe market={market} horizon={horizon} after start={start}"
        )
    return universe.sort_values(["PERMNO", "Date"], kind="mergesort").reset_index(drop=True)


def load_stw_universe_table(
    path: Path,
    *,
    start: str | None,
    permno_limit: int | None,
) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"universe panel not found: {path}")
    universe = pd.read_parquet(path)
    missing = [c for c in ("PERMNO", "Date") if c not in universe.columns]
    if missing:
        raise KeyError(f"universe missing required columns {missing}")
    universe = universe.copy()
    universe["Date"] = pd.to_datetime(universe["Date"])
    universe["PERMNO"] = universe["PERMNO"].astype("int64")
    if start is not None:
        universe = universe[universe["Date"] >= pd.Timestamp(start)].copy()
    if permno_limit is not None:
        keep = sorted(universe["PERMNO"].unique())[:permno_limit]
        universe = universe[universe["PERMNO"].isin(keep)].copy()
    if universe.empty:
        raise ValueError("empty universe after filters")
    return universe.sort_values(["PERMNO", "Date"], kind="mergesort").reset_index(drop=True)


def stw_weight_schemes(panel: pd.DataFrame, spec: MarketSpec) -> dict[str, str | None]:
    schemes: dict[str, str | None] = {"equal": None}
    for scheme in ("float", "total"):
        col = spec.cap_col(scheme)
        if col is None:
            continue
        if col not in panel.columns:
            raise KeyError(f"panel missing cap column {col} for weight scheme={scheme}")
        schemes[scheme] = col
    return schemes


def figure8_cnn_xlsx_path(
    market: str,
    horizon: int,
    *,
    cnn_backtest_root: Path = BACKTEST_CNN_ROOT,
    eval_horizon: int = 1,
    image_days: int = STW_FIGURE8_IMAGE_DAYS,
) -> Path:
    tag = model_run_tag(image_days, horizon)
    freq_dir = HORIZON_BACKTEST_DIR[horizon]
    stem = backtest_output_stem(tag, eval_horizon)
    return Path(cnn_backtest_root) / market / freq_dir / f"{stem}.xlsx"


def _hl_sharpe_from_h1_sheet(path: Path, sheet_name: str, tag: str) -> float:
    """Parse step-04 H1Perf Excel: row0=metric, row1=sig_rank, col0=config tag."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    metrics = raw.iloc[0, 1:].ffill()
    ranks = raw.iloc[1, 1:]
    dh_cols = [i for i, (m, r) in enumerate(zip(metrics, ranks)) if m == "Sharpe Ratio" and r == "DH"]
    if not dh_cols:
        raise KeyError(f"H1 sheet {sheet_name!r} missing Sharpe Ratio / DH column")
    col_idx = dh_cols[0] + 1
    tag_rows = raw.index[raw.iloc[:, 0].astype(str) == tag]
    if len(tag_rows) == 0:
        raise KeyError(f"tag {tag!r} not in H1 sheet {sheet_name!r}")
    return float(raw.iloc[int(tag_rows[0]), col_idx])


def load_figure8_cnn_sharpes(
    market: str,
    horizon: int,
    *,
    cnn_backtest_root: Path = BACKTEST_CNN_ROOT,
    eval_horizon: int = 1,
    image_days: int = STW_FIGURE8_IMAGE_DAYS,
    log=print,
) -> dict[str, float]:
    """Read step-04 I20 H1 Excel DH Sharpe ratios for Figure-8 red lines."""
    path = figure8_cnn_xlsx_path(
        market,
        horizon,
        cnn_backtest_root=cnn_backtest_root,
        eval_horizon=eval_horizon,
        image_days=image_days,
    )
    if not path.is_file():
        log(f"skip CNN Sharpe auto-load (missing): {path}")
        return {}

    tag = model_run_tag(image_days, horizon)
    out: dict[str, float] = {}
    for scheme in BACKTEST_WEIGHT_SCHEMES:
        out[scheme] = _hl_sharpe_from_h1_sheet(path, scheme, tag)
    log(f"loaded CNN H-L Sharpes from {path}: {out}")
    return out
