"""CLI: OHLC parquet + feature panel -> yearly image memmaps and label feathers.

Drawing uses OHLC only. Labels are joined from the feature panel at as_of after each image.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool, Value
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RESERVE_GIB = 16.0
MEM_PER_WORKER_GIB = 1.5
PROGRESS_EVERY = 500

from config import FEATURES_PARQUET, IMAGE_FREQ_DIR, IMAGES_ROOT, OHLC_PARQUET, WINDOW_DAYS
from data.calendar import as_of_dates_for_window
from data.images import try_build_window
from data.labels import labels_for_as_of
from data.parquet_io import load_calendar, permno_list, read_stock, read_stock_features
from utils.checkpoint import (
    append_image_labels,
    append_permno_checkpoint,
    load_permno_checkpoint,
    read_image_label_sidecars,
)
from utils.workers import resolve_workers

_PROGRESS = None
_TOTAL_STOCKS = None
_SKIP_STOCKS = None


def _init_progress(progress, total_stocks: int, skip_stocks: int) -> None:
    global _PROGRESS, _TOTAL_STOCKS, _SKIP_STOCKS
    _PROGRESS = progress
    _TOTAL_STOCKS = total_stocks
    _SKIP_STOCKS = skip_stocks


def log(msg: str) -> None:
    print(msg, flush=True)


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
    (permno_chunk, ohlc_path, features_path, calendar, window_days_list,
     as_of_per_window, worker_id, tmp_root, output_root) = task

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
            mode = "ab" if path.exists() else "wb"
            handles[key] = open(path, mode)
            labels[key] = []
        return handles[key]

    for i, permno in enumerate(permno_chunk):
        permno = int(permno)
        stock_df = read_stock(ohlc_path, permno)
        stock_dates = set(stock_df["DlyCalDt"])
        feature_panel = read_stock_features(features_path, permno)
        for window_days in window_days_list:
            for as_of in as_of_per_window[window_days]:
                if as_of not in stock_dates:
                    continue
                built = try_build_window(
                    stock_df=stock_df,
                    permno=permno,
                    as_of=as_of,
                    window_days=window_days,
                    calendar_last=calendar_last,
                )
                if built is None:
                    continue
                image, _meta = built
                shard(window_days, as_of.year).write(image.astype(np.uint8).tobytes())
                row = labels_for_as_of(feature_panel, as_of, permno, window_days)
                labels[(window_days, as_of.year)].append(row)
                append_image_labels(worker_dir, window_days, as_of.year, [row])
        append_permno_checkpoint(output_root, permno)
        with _PROGRESS.get_lock():
            _PROGRESS.value += 1
            done = _PROGRESS.value
        if done % PROGRESS_EVERY == 0 or done == _TOTAL_STOCKS:
            pct = 100.0 * done / _TOTAL_STOCKS
            log(
                f"progress {pct:5.1f}%  stocks {done}/{_TOTAL_STOCKS}  "
                f"skip={_SKIP_STOCKS}  worker={worker_id} chunk={i + 1}/{len(permno_chunk)}"
            )

    for h in handles.values():
        h.close()
    return labels, shard_paths


def build_window_images(
    ohlc_path: Path = OHLC_PARQUET,
    features_path: Path = FEATURES_PARQUET,
    output_root: Path = IMAGES_ROOT,
    window_days_list: tuple[int, ...] = WINDOW_DAYS,
    permno_limit: int | None = None,
    n_workers: int | None = None,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
    fresh: bool = False,
) -> None:
    if not ohlc_path.exists():
        raise FileNotFoundError(f"OHLC parquet not found: {ohlc_path}; run 01_prepare_data ohlc first")
    if not features_path.exists():
        raise FileNotFoundError(
            f"features parquet not found: {features_path}; run 01_prepare_data features first"
        )

    if fresh and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    calendar = load_calendar(ohlc_path)
    permnos = permno_list(ohlc_path)
    if permno_limit is not None:
        permnos = permnos[:permno_limit]

    done_permnos = load_permno_checkpoint(output_root)
    pending = np.array([p for p in permnos if int(p) not in done_permnos], dtype=np.int64)
    skip_stocks = len(permnos) - len(pending)
    log(f"images resume: skip={skip_stocks} pending={len(pending)} total={len(permnos)}")

    as_of_per_window = {w: as_of_dates_for_window(w, calendar) for w in window_days_list}
    years_per_window = {
        w: sorted({a.year for a in as_of_per_window[w]}) for w in window_days_list
    }

    tmp_root = output_root / ".tmp_workers"
    if len(pending) == 0:
        if not tmp_root.exists():
            log(f"skip images (checkpoint): {output_root}")
            return
        log("all permnos checkpointed; merging year bundles")
    else:
        n_workers, diag = resolve_workers(
            len(pending),
            override=n_workers,
            reserve_gib=reserve_gib,
            mem_per_worker_gib=MEM_PER_WORKER_GIB,
        )
        log(f"workers={n_workers} diag={diag}")

        tmp_root.mkdir(parents=True, exist_ok=True)

        progress = Value("i", 0)
        total_stocks = len(pending)
        chunks = chunk_permnos(pending, n_workers)
        tasks = [
            (chunk, ohlc_path, features_path, calendar, window_days_list,
             as_of_per_window, wid, tmp_root, output_root)
            for wid, chunk in enumerate(chunks)
        ]

        if n_workers == 1:
            _init_progress(progress, total_stocks, skip_stocks)
            process_permno_chunk(tasks[0])
        else:
            with Pool(
                n_workers,
                initializer=_init_progress,
                initargs=(progress, total_stocks, skip_stocks),
            ) as pool:
                pool.map(process_permno_chunk, tasks)

    for window_days in window_days_list:
        freq_dir = output_root / IMAGE_FREQ_DIR[window_days]
        freq_dir.mkdir(parents=True, exist_ok=True)
        for year in years_per_window[window_days]:
            final_dat = freq_dir / f"{window_days}d_{year}_images.dat"
            all_labels = read_image_label_sidecars(tmp_root, window_days, year)
            with open(final_dat, "wb") as out:
                for worker_dir in sorted(tmp_root.glob("worker_*")):
                    shard = worker_dir / IMAGE_FREQ_DIR[window_days] / f"{window_days}d_{year}_images.dat"
                    if shard.is_file():
                        with open(shard, "rb") as f:
                            out.write(f.read())
            write_year_bundle(freq_dir, window_days, year, final_dat, all_labels)

    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    log(f"done: stocks={len(permnos)} skip={skip_stocks} -> {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate price-trend images and label feathers")
    parser.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    parser.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    parser.add_argument("--output", type=Path, default=IMAGES_ROOT)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOW_DAYS))
    parser.add_argument("--permno-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    parser.add_argument("--fresh", action="store_true", help="delete images and rebuild")
    args = parser.parse_args()

    build_window_images(
        ohlc_path=args.parquet,
        features_path=args.features,
        output_root=args.output,
        window_days_list=tuple(args.windows),
        permno_limit=args.permno_limit,
        n_workers=args.workers,
        reserve_gib=args.reserve_gib,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
