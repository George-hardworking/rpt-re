"""Memmap image bundles + feather labels for CNN training."""

from __future__ import annotations

import fcntl
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    IMAGES_ROOT,
    SAMPLE_END,
    TEST_START,
    TRAIN_END,
    TRAIN_VAL_FRAC,
    image_bundle_dir,
    sample_freq_for_horizon,
)
from data.image_store import (
    bytes_per_image_for,
    list_image_years as _list_image_years,
)
from data.images import image_shape
from viz.preview_images import load_memmap_images, memmap_image_path


PIXEL_STATS_CHUNK = 4096


@dataclass(frozen=True)
class SampleRef:
    year: int
    row_idx: int
    label: int
    date: pd.Timestamp
    permno: int


@dataclass
class SampleTable:
    years: np.ndarray
    row_idx: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    permnos: np.ndarray

    def __len__(self) -> int:
        return int(self.years.shape[0])

    def take(self, idx: np.ndarray) -> SampleTable:
        return SampleTable(
            years=self.years[idx],
            row_idx=self.row_idx[idx],
            labels=self.labels[idx],
            dates=self.dates[idx],
            permnos=self.permnos[idx],
        )


def list_image_years(
    images_root: Path,
    window_days: int,
    sample_freq: str,
) -> list[int]:
    return _list_image_years(images_root, window_days, sample_freq)


def label_column(horizon: int) -> str:
    return f"Ret_{horizon}d_label"


def load_year_labels(
    images_root: Path,
    window_days: int,
    year: int,
    sample_freq: str,
) -> pd.DataFrame:
    freq_dir = images_root / image_bundle_dir(window_days, sample_freq)
    feather_path = freq_dir / f"{window_days}d_{year}_labels.feather"
    return pd.read_feather(feather_path)


def build_year_table(
    images_root: Path,
    window_days: int,
    horizon: int,
    year: int,
    sample_freq: str,
) -> SampleTable:
    labels = load_year_labels(images_root, window_days, year, sample_freq)
    col = label_column(horizon)
    dat_path = memmap_image_path(
        images_root, window_days, year, sample_freq=sample_freq
    )
    n_images = Path(dat_path).stat().st_size // bytes_per_image_for(dat_path, window_days)
    n_labels = len(labels)
    if n_labels < n_images:
        raise ValueError(
            f"label/image count mismatch window={window_days} freq={sample_freq} "
            f"year={year}: {n_labels} labels vs {n_images} images"
        )
    if n_labels > n_images:
        labels = labels.iloc[:n_images]
    # Missing forward return is label 2; keep original memmap row index.
    mask = labels[col].isin((0, 1)).to_numpy()
    row_idx = np.flatnonzero(mask).astype(np.int32)
    kept = labels.iloc[row_idx]
    n = len(kept)
    return SampleTable(
        years=np.full(n, year, dtype=np.int32),
        row_idx=row_idx,
        labels=kept[col].to_numpy(dtype=np.int8),
        dates=pd.to_datetime(kept["Date"]).to_numpy(),
        permnos=kept["StockID"].to_numpy(dtype=np.int64),
    )


def _concat_tables(parts: list[SampleTable]) -> SampleTable:
    return SampleTable(
        years=np.concatenate([p.years for p in parts]),
        row_idx=np.concatenate([p.row_idx for p in parts]),
        labels=np.concatenate([p.labels for p in parts]),
        dates=np.concatenate([p.dates for p in parts]),
        permnos=np.concatenate([p.permnos for p in parts]),
    )


def collect_samples(
    images_root: Path,
    window_days: int,
    horizon: int,
    *,
    sample_freq: str | None = None,
    year_limit: int | None = None,
    max_samples: int | None = None,
) -> SampleTable:
    freq = sample_freq if sample_freq is not None else sample_freq_for_horizon(horizon)
    years = list_image_years(images_root, window_days, freq)
    if year_limit is not None:
        years = years[:year_limit]
    parts: list[SampleTable] = []
    n = 0
    for year in years:
        part = build_year_table(images_root, window_days, horizon, year, freq)
        parts.append(part)
        n += len(part)
        if max_samples is not None and n >= max_samples:
            return _concat_tables(parts).take(np.arange(max_samples, dtype=np.int64))
    return _concat_tables(parts)


def split_samples(
    samples: SampleTable,
    *,
    seed: int,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    sample_end: str = SAMPLE_END,
    train_val_frac: float = TRAIN_VAL_FRAC,
) -> tuple[SampleTable, SampleTable, SampleTable]:
    dates = pd.DatetimeIndex(samples.dates)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    sample_end_ts = pd.Timestamp(sample_end)

    pool_idx = np.flatnonzero(dates <= train_end_ts)
    test_idx = np.flatnonzero((dates >= test_start_ts) & (dates <= sample_end_ts))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool_idx))
    n_train = int(len(pool_idx) * train_val_frac)
    train_idx = pool_idx[order[:n_train]]
    val_idx = pool_idx[order[n_train:]]
    return samples.take(train_idx), samples.take(val_idx), samples.take(test_idx)


def compute_pixel_mean_std(
    images_root: Path,
    window_days: int,
    sample_freq: str,
    years: np.ndarray,
    row_idx: np.ndarray,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    if len(years) == 0:
        raise ValueError("cannot compute pixel stats on empty sample list")
    mmap_cache: dict[int, np.ndarray] = {}
    total_sum = np.int64(0)
    total_sq = np.int64(0)
    count = np.int64(0)
    unique_years = np.unique(years)
    n_years = len(unique_years)
    for year_idx, year in enumerate(unique_years):
        year_i = int(year)
        if year_i not in mmap_cache:
            dat_path = memmap_image_path(
                images_root, window_days, year_i, sample_freq=sample_freq
            )
            mmap_cache[year_i] = load_memmap_images(dat_path, window_days)
        sel = years == year
        year_rows = row_idx[sel]
        n_rows = len(year_rows)
        for start in range(0, n_rows, PIXEL_STATS_CHUNK):
            end = min(start + PIXEL_STATS_CHUNK, n_rows)
            chunk_rows = year_rows[start:end]
            imgs = mmap_cache[year_i][chunk_rows]
            imgs_i64 = imgs.astype(np.int64)
            total_sum += imgs_i64.sum(dtype=np.int64)
            total_sq += np.square(imgs_i64, dtype=np.int64).sum(dtype=np.int64)
            count += np.int64(imgs.size)
            if log_fn is not None and n_rows > PIXEL_STATS_CHUNK:
                pct = 100.0 * end / n_rows
                log_fn(
                    f"pixel stats: year={year_i} {pct:.0f}% "
                    f"({year_idx + 1}/{n_years} years)"
                )
        if log_fn is not None:
            log_fn(f"pixel stats: year={year_i} done ({year_idx + 1}/{n_years} years)")
    count_f = float(count)
    mean = float(total_sum) / count_f
    var = float(total_sq) / count_f - mean * mean
    if var < 0.0:
        var = 0.0
    std = math.sqrt(var)
    if std == 0.0:
        std = 1.0
    return mean, std


def pixel_stats_cache_path(
    models_root: Path,
    window_days: int,
    horizon: int,
    split_seed: int,
) -> Path:
    return models_root / f"I{window_days}_R{horizon}" / f"pixel_stats_split{split_seed}.npz"


def load_or_compute_pixel_stats(
    images_root: Path,
    window_days: int,
    horizon: int,
    train: SampleTable,
    *,
    sample_freq: str,
    split_seed: int,
    cache_path: Path | None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    if cache_path is None:
        return compute_pixel_mean_std(
            images_root,
            window_days,
            sample_freq,
            train.years,
            train.row_idx,
            log_fn=log_fn,
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if cache_path.is_file():
            payload = np.load(cache_path)
            n_train = int(payload["n_train"])
            if n_train != len(train):
                raise ValueError(
                    f"pixel stats cache n_train={n_train} != {len(train)}: {cache_path}"
                )
            return float(payload["mean"]), float(payload["std"])
        mean, std = compute_pixel_mean_std(
            images_root,
            window_days,
            sample_freq,
            train.years,
            train.row_idx,
            log_fn=log_fn,
        )
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            mean=np.float64(mean),
            std=np.float64(std),
            n_train=np.int64(len(train)),
            image_days=np.int32(window_days),
            horizon=np.int32(horizon),
            split_seed=np.int32(split_seed),
        )
        tmp.replace(cache_path)
        return mean, std


class ImageLabelDataset(Dataset):
    def __init__(
        self,
        images_root: Path,
        window_days: int,
        samples: SampleTable,
        pixel_mean: float,
        pixel_std: float,
        *,
        sample_freq: str,
        mmap_cache: dict[int, np.ndarray] | None = None,
    ) -> None:
        self.images_root = images_root
        self.window_days = window_days
        self.sample_freq = sample_freq
        self.table = samples
        self.pixel_mean = float(pixel_mean)
        self.pixel_std = float(pixel_std)
        self._mmap_cache: dict[int, np.ndarray] = mmap_cache if mmap_cache is not None else {}
        for year in np.unique(samples.years):
            self._get_year_mmap(int(year))

    def __len__(self) -> int:
        return len(self.table)

    def _get_year_mmap(self, year: int) -> np.ndarray:
        if year not in self._mmap_cache:
            dat_path = memmap_image_path(
                self.images_root,
                self.window_days,
                year,
                sample_freq=self.sample_freq,
            )
            self._mmap_cache[year] = load_memmap_images(dat_path, self.window_days)
        return self._mmap_cache[year]

    def gather_uint8(self, indices: np.ndarray) -> np.ndarray:
        years = self.table.years[indices]
        rows = self.table.row_idx[indices]
        height, width = image_shape(self.window_days)
        imgs = np.empty((len(indices), height, width), dtype=np.uint8)
        for year in np.unique(years):
            sel = years == year
            imgs[sel] = self._get_year_mmap(int(year))[rows[sel]]
        return imgs

    def batch_xy(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        imgs = self.gather_uint8(indices)
        x = torch.from_numpy((imgs.astype(np.float32) - self.pixel_mean) / self.pixel_std)
        x = x.unsqueeze(1)
        y = torch.from_numpy(self.table.labels[indices].astype(np.int64))
        return x, y

    def batch_refs(self, indices: np.ndarray) -> list[SampleRef]:
        years = self.table.years
        rows = self.table.row_idx
        labels = self.table.labels
        dates = self.table.dates
        permnos = self.table.permnos
        return [
            SampleRef(
                year=int(years[i]),
                row_idx=int(rows[i]),
                label=int(labels[i]),
                date=pd.Timestamp(dates[i]),
                permno=int(permnos[i]),
            )
            for i in indices
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, SampleRef]:
        idx = np.asarray([index], dtype=np.int64)
        x, y = self.batch_xy(idx)
        ref = self.batch_refs(idx)[0]
        return x[0], y[0], ref


def model_run_tag(
    image_days: int,
    horizon: int,
    *,
    init_from_image_days: int | None = None,
) -> str:
    tag = f"I{image_days}_R{horizon}"
    if init_from_image_days is not None:
        tag = f"{tag}_fromI{init_from_image_days}"
    return tag


def model_run_dir(
    models_root: Path,
    image_days: int,
    horizon: int,
    seed: int,
    *,
    init_from_image_days: int | None = None,
) -> Path:
    return (
        models_root
        / model_run_tag(image_days, horizon, init_from_image_days=init_from_image_days)
        / f"seed{seed}"
    )


def default_images_root() -> Path:
    return IMAGES_ROOT
