"""CLI: CSV -> OHLC parquet -> daily feature panel.

Subcommands: ohlc (CSV ingest), features (derived labels), all (both).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from multiprocessing import Pool, Value
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_RESERVE_GIB = 16.0
MEM_PER_WORKER_GIB = 0.8
PROGRESS_EVERY = 500

from config import FEATURES_PARQUET, OHLC_PARQUET, PROCESSED_DIR, RAW_OHLC_CSV
from data.labels import build_stock_features
from data.parquet_io import permno_list, prepare_ohlc_parquet, read_stock
from utils.checkpoint import features_partition_complete, write_features_partition
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


def process_permno_chunk(task: tuple) -> tuple[int, int]:
    permno_chunk, ohlc_path, features_path, worker_id = task
    built = 0
    skipped = 0
    for i, permno in enumerate(permno_chunk):
        permno = int(permno)
        if features_partition_complete(features_path, permno):
            skipped += 1
        else:
            stock_df = read_stock(ohlc_path, permno)
            features = build_stock_features(stock_df, permno)
            table = pa.Table.from_pandas(features, preserve_index=False)
            write_features_partition(features_path, table, permno)
            built += 1
        with _PROGRESS.get_lock():
            _PROGRESS.value += 1
            done = _PROGRESS.value
        if done % PROGRESS_EVERY == 0 or done == _TOTAL_STOCKS:
            pct = 100.0 * done / _TOTAL_STOCKS
            log(
                f"progress {pct:5.1f}%  stocks {done}/{_TOTAL_STOCKS}  "
                f"skip={_SKIP_STOCKS}  worker={worker_id} "
                f"chunk={i + 1}/{len(permno_chunk)}"
            )
    return built, skipped


def build_features(
    ohlc_path: Path = OHLC_PARQUET,
    output_path: Path = FEATURES_PARQUET,
    permno_limit: int | None = None,
    n_workers: int | None = None,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
    fresh: bool = False,
) -> Path:
    if not ohlc_path.exists():
        raise FileNotFoundError(f"OHLC parquet not found: {ohlc_path}; run ohlc first")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if fresh and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for stale in output_path.glob(".tmp_PERMNO=*"):
        if stale.is_dir():
            shutil.rmtree(stale)

    permnos = permno_list(ohlc_path)
    if permno_limit is not None:
        permnos = permnos[:permno_limit]

    skip_stocks = sum(1 for p in permnos if features_partition_complete(output_path, int(p)))
    pending = sum(1 for p in permnos if not features_partition_complete(output_path, int(p)))
    log(f"features resume: skip={skip_stocks} pending={pending} total={len(permnos)}")

    n_workers, diag = resolve_workers(
        len(permnos),
        override=n_workers,
        reserve_gib=reserve_gib,
        mem_per_worker_gib=MEM_PER_WORKER_GIB,
    )
    log(f"workers={n_workers} diag={diag}")

    progress = Value("i", 0)
    total_stocks = len(permnos)
    chunks = chunk_permnos(permnos, n_workers)
    tasks = [
        (chunk, ohlc_path, output_path, wid)
        for wid, chunk in enumerate(chunks)
    ]

    if n_workers == 1:
        _init_progress(progress, total_stocks, skip_stocks)
        built, skipped = process_permno_chunk(tasks[0])
    else:
        with Pool(
            n_workers,
            initializer=_init_progress,
            initargs=(progress, total_stocks, skip_stocks),
        ) as pool:
            chunk_stats = pool.map(process_permno_chunk, tasks)
        built = sum(b for b, _ in chunk_stats)
        skipped = sum(s for _, s in chunk_stats)

    log(
        f"done: n_workers={n_workers} stocks={total_stocks} "
        f"built={built} skip={skipped} -> {output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OHLC parquet and daily feature panel")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ohlc = sub.add_parser("ohlc", help="CSV -> partitioned OHLC parquet")
    p_ohlc.add_argument("--raw", type=Path, default=RAW_OHLC_CSV)
    p_ohlc.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_ohlc.add_argument("--fresh", action="store_true", help="rebuild OHLC from CSV")

    p_features = sub.add_parser("features", help="OHLC parquet -> feature panel")
    p_features.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_features.add_argument("--output", type=Path, default=FEATURES_PARQUET)
    p_features.add_argument("--permno-limit", type=int, default=None)
    p_features.add_argument("--workers", type=int, default=None)
    p_features.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    p_features.add_argument("--fresh", action="store_true", help="delete features and rebuild")

    p_all = sub.add_parser("all", help="ohlc then features")
    p_all.add_argument("--raw", type=Path, default=RAW_OHLC_CSV)
    p_all.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_all.add_argument("--output", type=Path, default=FEATURES_PARQUET)
    p_all.add_argument("--permno-limit", type=int, default=None)
    p_all.add_argument("--workers", type=int, default=None)
    p_all.add_argument("--reserve-gib", type=float, default=DEFAULT_RESERVE_GIB)
    p_all.add_argument("--fresh", action="store_true", help="rebuild OHLC and/or features")

    args = parser.parse_args()

    if args.command == "ohlc":
        path = prepare_ohlc_parquet(args.raw, args.parquet, fresh=args.fresh, log=log)
        log(f"prepared {path}")
    elif args.command == "features":
        build_features(
            ohlc_path=args.parquet,
            output_path=args.output,
            permno_limit=args.permno_limit,
            n_workers=args.workers,
            reserve_gib=args.reserve_gib,
            fresh=args.fresh,
        )
    elif args.command == "all":
        path = prepare_ohlc_parquet(args.raw, args.parquet, fresh=args.fresh, log=log)
        log(f"prepared {path}")
        build_features(
            ohlc_path=args.parquet,
            output_path=args.output,
            permno_limit=args.permno_limit,
            n_workers=args.workers,
            reserve_gib=args.reserve_gib,
            fresh=args.fresh,
        )


if __name__ == "__main__":
    main()
