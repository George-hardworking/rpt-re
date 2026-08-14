"""Memmap image bundles + feather labels for CNN training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    IMAGE_FREQ_DIR,
    IMAGES_ROOT,
    SAMPLE_END,
    TEST_START,
    TRAIN_END,
    TRAIN_VAL_FRAC,
)
from data.images import image_shape
from viz.preview_images import load_memmap_images, memmap_image_path


@dataclass(frozen=True)
class SampleRef:
    year: int
    row_idx: int
    label: int
    date: pd.Timestamp
    permno: int


def list_image_years(images_root: Path, window_days: int) -> list[int]:
    freq_dir = images_root / IMAGE_FREQ_DIR[window_days]
    years: list[int] = []
    for path in sorted(freq_dir.glob(f"{window_days}d_*_images.dat")):
        parts = path.stem.split("_")
        years.append(int(parts[1]))
    return years


def label_column(horizon: int) -> str:
    return f"Ret_{horizon}d_label"


def load_year_labels(images_root: Path, window_days: int, year: int) -> pd.DataFrame:
    freq_dir = images_root / IMAGE_FREQ_DIR[window_days]
    feather_path = freq_dir / f"{window_days}d_{year}_labels.feather"
    return pd.read_feather(feather_path)


def build_year_samples(
    images_root: Path,
    window_days: int,
    horizon: int,
    year: int,
) -> list[SampleRef]:
    labels = load_year_labels(images_root, window_days, year)
    col = label_column(horizon)
    valid = labels[col].isin((0, 1))
    labels = labels.loc[valid].reset_index(drop=True)
    dat_path = memmap_image_path(images_root, window_days, year)
    height, width = image_shape(window_days)
    n_images = Path(dat_path).stat().st_size // (height * width)
    if len(labels) != n_images:
        raise ValueError(
            f"label/image count mismatch window={window_days} year={year}: "
            f"{len(labels)} labels vs {n_images} images"
        )
    out: list[SampleRef] = []
    for row_idx, row in labels.iterrows():
        out.append(
            SampleRef(
                year=year,
                row_idx=int(row_idx),
                label=int(row[col]),
                date=pd.Timestamp(row["Date"]),
                permno=int(row["StockID"]),
            )
        )
    return out


def collect_samples(
    images_root: Path,
    window_days: int,
    horizon: int,
    *,
    year_limit: int | None = None,
    max_samples: int | None = None,
) -> list[SampleRef]:
    years = list_image_years(images_root, window_days)
    if year_limit is not None:
        years = years[:year_limit]
    samples: list[SampleRef] = []
    for year in years:
        samples.extend(build_year_samples(images_root, window_days, horizon, year))
        if max_samples is not None and len(samples) >= max_samples:
            return samples[:max_samples]
    return samples


def split_samples(
    samples: list[SampleRef],
    *,
    seed: int,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    sample_end: str = SAMPLE_END,
    train_val_frac: float = TRAIN_VAL_FRAC,
) -> tuple[list[SampleRef], list[SampleRef], list[SampleRef]]:
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    sample_end_ts = pd.Timestamp(sample_end)

    train_pool = [s for s in samples if s.date <= train_end_ts]
    test = [s for s in samples if test_start_ts <= s.date <= sample_end_ts]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_pool))
    n_train = int(len(train_pool) * train_val_frac)
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    train = [train_pool[int(i)] for i in train_idx]
    val = [train_pool[int(i)] for i in val_idx]
    return train, val, test


def compute_pixel_mean_std(
    images_root: Path,
    window_days: int,
    samples: list[SampleRef],
) -> tuple[float, float]:
    if not samples:
        raise ValueError("cannot compute pixel stats on empty sample list")
    mmap_cache: dict[int, np.memmap] = {}
    total_sum = 0.0
    total_sq = 0.0
    count = 0
    for ref in samples:
        if ref.year not in mmap_cache:
            dat_path = memmap_image_path(images_root, window_days, ref.year)
            mmap_cache[ref.year] = load_memmap_images(dat_path, window_days)
        img = mmap_cache[ref.year][ref.row_idx].astype(np.float64)
        total_sum += float(img.sum())
        total_sq += float((img * img).sum())
        count += img.size
    mean = total_sum / count
    var = total_sq / count - mean * mean
    std = math.sqrt(var)
    if std == 0.0:
        std = 1.0
    return mean, std


class ImageLabelDataset(Dataset):
    def __init__(
        self,
        images_root: Path,
        window_days: int,
        samples: list[SampleRef],
        pixel_mean: float,
        pixel_std: float,
    ) -> None:
        self.images_root = images_root
        self.window_days = window_days
        self.samples = samples
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self._mmap_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _get_year_mmap(self, year: int) -> np.ndarray:
        if year not in self._mmap_cache:
            dat_path = memmap_image_path(self.images_root, self.window_days, year)
            self._mmap_cache[year] = load_memmap_images(dat_path, self.window_days)
        return self._mmap_cache[year]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, SampleRef]:
        ref = self.samples[index]
        image = self._get_year_mmap(ref.year)[ref.row_idx].astype(np.float32)
        image = (image - self.pixel_mean) / self.pixel_std
        x = torch.from_numpy(image).unsqueeze(0)
        y = torch.tensor(ref.label, dtype=torch.long)
        return x, y, ref


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
    return models_root / model_run_tag(
        image_days, horizon, init_from_image_days=init_from_image_days
    ) / f"seed{seed}"


def default_images_root() -> Path:
    return IMAGES_ROOT
