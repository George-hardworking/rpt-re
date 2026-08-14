"""Checkpoint markers and resume helpers for long-running pipeline scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    IMAGES_CHECKPOINT_PAPER_CROSS,
    PAPER_CROSS_BUNDLES,
    WINDOW_DAYS,
    diagonal_bundles,
)

OHLC_COMPLETE = ".complete"
FEATURES_TMP_PREFIX = ".tmp_PERMNO="
IMAGES_CHECKPOINT_PERMNOS = ".checkpoint_permnos"

ImageBundle = tuple[int, str]


def _bundle_tag(window_days: int, sample_freq: str) -> str:
    return f"{window_days}{sample_freq}"


def images_checkpoint_name(
    bundles: tuple[ImageBundle, ...],
    *,
    all_diagonal: tuple[ImageBundle, ...] | None = None,
) -> str:
    """Checkpoint file name for a bundle set; diagonal full run uses `.checkpoint_permnos`."""
    if all_diagonal is None:
        all_diagonal = diagonal_bundles()
    if tuple(bundles) == all_diagonal:
        return IMAGES_CHECKPOINT_PERMNOS
    if tuple(bundles) == PAPER_CROSS_BUNDLES:
        return IMAGES_CHECKPOINT_PAPER_CROSS
    tag = "_".join(_bundle_tag(w, f) for w, f in bundles)
    return f"{IMAGES_CHECKPOINT_PERMNOS}_{tag}"


def images_checkpoint_name_windows(
    window_days_list: tuple[int, ...],
    all_windows: tuple[int, ...] = WINDOW_DAYS,
) -> str:
    """Legacy helper: diagonal bundles for the given window subset."""
    from config import windows_to_diagonal_bundles

    all_diag = windows_to_diagonal_bundles(all_windows)
    bundles = windows_to_diagonal_bundles(window_days_list)
    return images_checkpoint_name(bundles, all_diagonal=all_diag)


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


def load_permno_checkpoint(
    output_root: Path, filename: str = IMAGES_CHECKPOINT_PERMNOS
) -> set[int]:
    path = output_root / filename
    if not path.is_file():
        return set()
    return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}


def append_permno_checkpoint(
    output_root: Path,
    permno: int,
    filename: str = IMAGES_CHECKPOINT_PERMNOS,
) -> None:
    path = output_root / filename
    with open(path, "a") as f:
        f.write(f"{permno}\n")
        f.flush()


class LabelJsonlWriter:
    """Buffered jsonl sidecar writer; one open handle per (window_days, sample_freq, year)."""

    def __init__(self, worker_dir: Path) -> None:
        self.worker_dir = worker_dir
        self._handles: dict[tuple[int, str, int], object] = {}

    def append(
        self,
        window_days: int,
        sample_freq: str,
        year: int,
        rows: list[dict],
    ) -> None:
        if not rows:
            return
        key = (window_days, sample_freq, year)
        if key not in self._handles:
            from config import image_bundle_dir

            bundle_dir = self.worker_dir / image_bundle_dir(window_days, sample_freq)
            bundle_dir.mkdir(parents=True, exist_ok=True)
            path = bundle_dir / f"{window_days}d_{year}_labels.jsonl"
            mode = "a" if path.exists() else "w"
            self._handles[key] = open(path, mode)
        handle = self._handles[key]
        for row in rows:
            handle.write(json.dumps(row, default=str))
            handle.write("\n")

    def flush_all(self) -> None:
        for handle in self._handles.values():
            handle.flush()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def read_image_label_sidecars(
    tmp_root: Path,
    window_days: int,
    sample_freq: str,
    year: int,
) -> list[dict]:
    from config import image_bundle_dir

    bundle_name = image_bundle_dir(window_days, sample_freq)
    rows: list[dict] = []
    for worker_dir in sorted(tmp_root.glob("worker_*")):
        path = worker_dir / bundle_name / f"{window_days}d_{year}_labels.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
