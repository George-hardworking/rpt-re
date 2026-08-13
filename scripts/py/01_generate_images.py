"""CLI: CSV -> parquet -> price trend image dataset (prepare / build / all).

All one-shot pipeline logic lives here (runners, multiprocessing worker, helpers).
src/ holds only reusable core primitives (images / calendar / parquet_io); config
constants live in src/config.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from multiprocessing import Pool, Value
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RESERVE_GIB = 16.0
MEM_PER_WORKER_GIB = 1.5
PROGRESS_EVERY = 500

from config import (
    CSV_CHUNKSIZE,
    FUTURE_HORIZONS,
    IMAGE_FREQ_DIR,
    IMAGES_ROOT,
    OHLC_DTYPES,
    OHLC_PARQUET,
    PREPARE_COLS,
    PROCESSED_DIR,
    RAW_OHLC_CSV,
    WINDOW_DAYS,
)
from data.calendar import as_of_dates_for_window
from data.images import try_build_window
from data.parquet_io import load_calendar, permno_list, read_stock
from utils.workers import resolve_workers

_PROGRESS = None
_TOTAL_STOCKS = None


def _init_progress(progress, total_stocks: int) -> None:
    global _PROGRESS, _TOTAL_STOCKS
    _PROGRESS = progress
    _TOTAL_STOCKS = total_stocks


def log(msg: str) -> None:
    print(msg, flush=True)


def prepare_ohlc_parquet(
    raw_path: Path = RAW_OHLC_CSV,
    output_path: Path = OHLC_PARQUET,
    chunksize: int = CSV_CHUNKSIZE,
) -> Path:
    """Convert raw CRSP OHLC CSV to PERMNO-partitioned parquet without filling missing values."""
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw OHLC CSV not found: {raw_path}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    n_rows = 0
    n_chunks = 0
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=OHLC_DTYPES):
        n_chunks += 1
        n_rows += len(chunk)
        chunk = chunk[PREPARE_COLS].copy()
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=["PERMNO"],
        )
        log(f"prepare chunk={n_chunks} rows={n_rows:,} -> {output_path}")

    return output_path


def chunk_permnos(seq: np.ndarray, n: int) -> list[np.ndarray]:
    k, m = divmod(len(seq), n)
    return [seq[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def write_year_bundle(
    output_dir: Path,
    window_days: int,
    year: int,
    dat_path: Path,
    labels: list[dict],
) -> None:
    if not labels:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    label_df = pd.DataFrame(labels)
    feather_path = output_dir / f"{window_days}d_{year}_labels.feather"
    label_df.to_feather(feather_path)
    log(f"window={window_days} year={year} images={len(labels)} -> {dat_path}")


def process_permno_chunk(task: tuple) -> tuple[dict, dict]:
    """Worker: read each stock once, render all windows, write per-worker .dat shards.

    Top-level so multiprocessing can pickle it by reference; the `if __name__ == "__main__"`
    guard keeps children from re-running the CLI under spawn.
    """
    (permno_chunk, parquet_path, calendar, window_days_list,
     as_of_per_window, future_horizons, worker_id, tmp_root) = task

    calendar_last = calendar[-1]
    worker_dir = tmp_root / f"worker_{worker_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[tuple[int, int], object] = {}
    shard_paths: dict[tuple[int, int], Path] = {}
    labels: dict[tuple[int, int], list[dict]] = {}

    def shard(window_days: int, year: int):
        key = (window_days, year)
        if key not in handles:
            freq_dir = worker_dir / IMAGE_FREQ_DIR[window_days]
            freq_dir.mkdir(parents=True, exist_ok=True)
            path = freq_dir / f"{window_days}d_{year}_images.dat"
            shard_paths[key] = path
            handles[key] = open(path, "wb")
            labels[key] = []
        return handles[key]

    for i, permno in enumerate(permno_chunk):
        stock_df = read_stock(parquet_path, int(permno))
        stock_dates = set(stock_df["DlyCalDt"])
        for window_days in window_days_list:
            for as_of in as_of_per_window[window_days]:
                if as_of not in stock_dates:
                    continue
                built = try_build_window(
                    stock_df=stock_df,
                    permno=int(permno),
                    as_of=as_of,
                    window_days=window_days,
                    calendar=calendar,
                    calendar_last=calendar_last,
                    future_horizons=future_horizons,
                )
                if built is None:
                    continue
                image, label = built
                shard(window_days, as_of.year).write(image.astype(np.uint8).tobytes())
                labels[(window_days, as_of.year)].append(label)
        with _PROGRESS.get_lock():
            _PROGRESS.value += 1
            done = _PROGRESS.value
        if done % PROGRESS_EVERY == 0 or done == _TOTAL_STOCKS:
            pct = 100.0 * done / _TOTAL_STOCKS
            log(
                f"progress {pct:5.1f}%  stocks {done}/{_TOTAL_STOCKS}  "
                f"worker={worker_id} chunk={i + 1}/{len(permno_chunk)}"
            )

    for h in handles.values():
        h.close()
    return labels, shard_paths


def build_window_images(
    parquet_path: Path = OHLC_PARQUET,
    output_root: Path = IMAGES_ROOT,
    window_days_list: tuple[int, ...] = WINDOW_DAYS,
    permno_limit: int | None = None,
    n_workers: int | None = None,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}; run prepare first")

    calendar = load_calendar(parquet_path)
    permnos = permno_list(parquet_path)
    if permno_limit is not None:
        permnos = permnos[:permno_limit]

    as_of_per_window = {w: as_of_dates_for_window(w, calendar) for w in window_days_list}
    years_per_window = {
        w: sorted({a.year for a in as_of_per_window[w]}) for w in window_days_list
    }

    n_workers, diag = resolve_workers(
        len(permnos),
        override=n_workers,
        reserve_gib=reserve_gib,
        mem_per_worker_gib=MEM_PER_WORKER_GIB,
    )
    log(f"workers={n_workers} diag={diag}")

    tmp_root = output_root / ".tmp_workers"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    progress = Value("i", 0)
    total_stocks = len(permnos)
    chunks = chunk_permnos(permnos, n_workers)
    tasks = [
        (chunk, parquet_path, calendar, window_days_list,
         as_of_per_window, FUTURE_HORIZONS, wid, tmp_root)
        for wid, chunk in enumerate(chunks)
    ]

    if n_workers == 1:
        _init_progress(progress, total_stocks)
        results = [process_permno_chunk(tasks[0])]
    else:
        with Pool(
            n_workers,
            initializer=_init_progress,
            initargs=(progress, total_stocks),
        ) as pool:
            results = pool.map(process_permno_chunk, tasks)

    for window_days in window_days_list:
        freq_dir = output_root / IMAGE_FREQ_DIR[window_days]
        freq_dir.mkdir(parents=True, exist_ok=True)
        for year in years_per_window[window_days]:
            key = (window_days, year)
            final_dat = freq_dir / f"{window_days}d_{year}_images.dat"
            all_labels: list[dict] = []
            with open(final_dat, "wb") as out:
                for labels, shard_paths in results:
                    shard = shard_paths.get(key)
                    if shard is None:
                        continue
                    with open(shard, "rb") as f:
                        out.write(f.read())
                    all_labels.extend(labels[key])
            write_year_bundle(freq_dir, window_days, year, final_dat, all_labels)

    shutil.rmtree(tmp_root)
    log(f"done: n_workers={n_workers} -> {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OHLC data preparation and image generation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="CSV -> partitioned parquet")
    p_prepare.add_argument("--raw", type=Path, default=RAW_OHLC_CSV)
    p_prepare.add_argument("--parquet", type=Path, default=OHLC_PARQUET)

    p_build = sub.add_parser("build", help="parquet -> yearly image memmaps + labels")
    p_build.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_build.add_argument("--output", type=Path, default=IMAGES_ROOT)
    p_build.add_argument("--windows", type=int, nargs="+", default=list(WINDOW_DAYS))
    p_build.add_argument("--permno-limit", type=int, default=None)
    p_build.add_argument("--workers", type=int, default=None, help="worker processes (default: memory-aware auto)")
    p_build.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB, help="RAM GiB to leave free when auto-sizing workers")

    p_all = sub.add_parser("all", help="prepare then build")
    p_all.add_argument("--raw", type=Path, default=RAW_OHLC_CSV)
    p_all.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_all.add_argument("--output", type=Path, default=IMAGES_ROOT)
    p_all.add_argument("--windows", type=int, nargs="+", default=list(WINDOW_DAYS))
    p_all.add_argument("--permno-limit", type=int, default=None)
    p_all.add_argument("--workers", type=int, default=None, help="worker processes (default: memory-aware auto)")
    p_all.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB, help="RAM GiB to leave free when auto-sizing workers")

    args = parser.parse_args()

    if args.command == "prepare":
        path = prepare_ohlc_parquet(args.raw, args.parquet)
        log(f"prepared {path}")
    elif args.command == "build":
        build_window_images(
            parquet_path=args.parquet,
            output_root=args.output,
            window_days_list=tuple(args.windows),
            permno_limit=args.permno_limit,
            n_workers=args.workers,
            reserve_gib=args.reserve_gib,
        )
    elif args.command == "all":
        path = prepare_ohlc_parquet(args.raw, args.parquet)
        log(f"prepared {path}")
        build_window_images(
            parquet_path=args.parquet,
            output_root=args.output,
            window_days_list=tuple(args.windows),
            permno_limit=args.permno_limit,
            n_workers=args.workers,
            reserve_gib=args.reserve_gib,
        )


if __name__ == "__main__":
    main()
