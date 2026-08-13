"""Build yearly image memmaps and label feather files from prepared OHLC parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FUTURE_HORIZONS,
    IMAGE_FREQ_DIR,
    IMAGES_ROOT,
    OHLC_PARQUET,
    WINDOW_DAYS,
)
from data.calendar import as_of_dates_for_window
from data.images import image_shape, try_build_window
from data.parquet_io import load_calendar, permno_list, read_stock


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
    print(f"window={window_days} year={year} images={len(labels)} -> {dat_path}")


def build_window_images(
    parquet_path: Path = OHLC_PARQUET,
    output_root: Path = IMAGES_ROOT,
    window_days_list: tuple[int, ...] = WINDOW_DAYS,
    permno_limit: int | None = None,
) -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}; run prepare first")

    calendar = load_calendar(parquet_path)
    calendar_last = calendar[-1]
    permnos = permno_list(parquet_path)
    if permno_limit is not None:
        permnos = permnos[:permno_limit]

    for window_days in window_days_list:
        as_of_dates = as_of_dates_for_window(window_days, calendar)
        years = sorted({as_of.year for as_of in as_of_dates})
        freq_dir = output_root / IMAGE_FREQ_DIR[window_days]
        freq_dir.mkdir(parents=True, exist_ok=True)

        year_labels: dict[int, list[dict]] = {year: [] for year in years}
        year_files: dict[int, Path] = {
            year: freq_dir / f"{window_days}d_{year}_images.dat" for year in years
        }
        year_handles: dict[int, object] = {
            year: open(year_files[year], "wb") for year in years
        }

        for i, permno in enumerate(permnos):
            stock_df = read_stock(parquet_path, int(permno))
            stock_dates = set(stock_df["DlyCalDt"])

            for as_of in as_of_dates:
                if as_of not in stock_dates:
                    continue
                built = try_build_window(
                    stock_df=stock_df,
                    permno=int(permno),
                    as_of=as_of,
                    window_days=window_days,
                    calendar=calendar,
                    calendar_last=calendar_last,
                    future_horizons=FUTURE_HORIZONS,
                )
                if built is None:
                    continue
                image, label = built
                year = as_of.year
                year_handles[year].write(image.astype(np.uint8).tobytes())
                year_labels[year].append(label)

            if (i + 1) % 500 == 0:
                print(f"window={window_days} processed {i + 1}/{len(permnos)} stocks")

        for year in years:
            year_handles[year].close()
            write_year_bundle(freq_dir, window_days, year, year_files[year], year_labels[year])
