"""Checkpoint markers and resume helpers for long-running pipeline scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OHLC_COMPLETE = ".complete"
FEATURES_TMP_PREFIX = ".tmp_PERMNO="
IMAGES_CHECKPOINT_PERMNOS = ".checkpoint_permnos"


def ohlc_is_complete(output_path: Path) -> bool:
    return (output_path / OHLC_COMPLETE).is_file()


def mark_ohlc_complete(output_path: Path) -> None:
    (output_path / OHLC_COMPLETE).touch()


def features_partition_complete(root: Path, permno: int) -> bool:
    part = root / f"PERMNO={permno}"
    return part.is_dir() and any(part.glob("*.parquet"))


def write_features_partition(root: Path, table: pa.Table, permno: int) -> None:
    tmp_root = root / f"{FEATURES_TMP_PREFIX}{permno}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    pq.write_to_dataset(
        table,
        root_path=str(tmp_root),
        partition_cols=["PERMNO"],
    )
    inner = tmp_root / f"PERMNO={permno}"
    final = root / f"PERMNO={permno}"
    if final.exists():
        shutil.rmtree(final)
    inner.rename(final)
    shutil.rmtree(tmp_root)


def load_permno_checkpoint(output_root: Path) -> set[int]:
    path = output_root / IMAGES_CHECKPOINT_PERMNOS
    if not path.is_file():
        return set()
    return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}


def append_permno_checkpoint(output_root: Path, permno: int) -> None:
    path = output_root / IMAGES_CHECKPOINT_PERMNOS
    with open(path, "a") as f:
        f.write(f"{permno}\n")
        f.flush()


def append_image_labels(worker_dir: Path, window_days: int, year: int, rows: list[dict]) -> None:
    if not rows:
        return
    path = worker_dir / f"{window_days}d_{year}_labels.jsonl"
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=str))
            f.write("\n")
        f.flush()


def read_image_label_sidecars(tmp_root: Path, window_days: int, year: int) -> list[dict]:
    rows: list[dict] = []
    for worker_dir in sorted(tmp_root.glob("worker_*")):
        path = worker_dir / f"{window_days}d_{year}_labels.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
