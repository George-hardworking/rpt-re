"""Preview OHLC images from parquet or saved memmap bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import IMAGE_FREQ_DIR, IMAGES_ROOT, OHLC_PARQUET, WINDOW_DAYS
from data.calendar import as_of_dates_for_window
from data.images import image_shape, try_build_window
from data.parquet_io import load_calendar, read_stock


def default_permno(parquet_path: Path) -> int:
    return int(pd.read_parquet(parquet_path, columns=["PERMNO"]).iloc[0]["PERMNO"])


def build_sample_windows(
    parquet_path: Path = OHLC_PARQUET,
    permno: int | None = None,
    window_days_list: tuple[int, ...] = WINDOW_DAYS,
    use_last_as_of: bool = True,
    as_of_map: dict[int, pd.Timestamp] | None = None,
) -> list[tuple[int, np.ndarray, dict[str, Any], pd.Timestamp]]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    calendar = load_calendar(parquet_path)
    calendar_last = calendar[-1]
    stock_id = permno if permno is not None else default_permno(parquet_path)
    stock_df = read_stock(parquet_path, stock_id)

    built: list[tuple[int, np.ndarray, dict[str, Any], pd.Timestamp]] = []
    for window_days in window_days_list:
        as_ofs = as_of_dates_for_window(window_days, calendar)
        if as_of_map and window_days in as_of_map:
            as_of = as_of_map[window_days]
        elif use_last_as_of:
            as_of = as_ofs[-1]
        else:
            as_of = as_ofs[0]

        result = try_build_window(
            stock_df=stock_df,
            permno=stock_id,
            as_of=as_of,
            window_days=window_days,
            calendar=calendar,
            calendar_last=calendar_last,
        )
        if result is None:
            raise ValueError(
                f"no image for PERMNO={stock_id} window={window_days} as_of={as_of}"
            )
        image, label = result
        built.append((window_days, image, label, as_of))
    return built


def plot_sample_windows(
    samples: list[tuple[int, np.ndarray, dict[str, Any], pd.Timestamp]],
    permno: int,
    figsize: tuple[float, float] = (12, 4),
) -> Any:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(samples), figsize=figsize)
    if len(samples) == 1:
        axes = [axes]
    for ax, (window_days, image, _label, as_of) in zip(axes, samples):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"I{window_days} PERMNO={permno} {as_of.date()}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def memmap_image_path(
    output_root: Path,
    window_days: int,
    year: int,
) -> Path:
    freq_dir = output_root / IMAGE_FREQ_DIR[window_days]
    return freq_dir / f"{window_days}d_{year}_images.dat"


def load_memmap_images(
    dat_path: Path,
    window_days: int,
    mode: str = "r",
) -> np.ndarray:
    height, width = image_shape(window_days)
    flat = np.memmap(dat_path, dtype=np.uint8, mode=mode)
    n_images = flat.size // (height * width)
    return flat.reshape(n_images, height, width)


def plot_memmap_index(
    dat_path: Path,
    window_days: int,
    index: int = 0,
    title: str | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    images = load_memmap_images(dat_path, window_days)
    if index < 0 or index >= len(images):
        raise IndexError(f"index {index} out of range for {len(images)} images")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
    ax.set_title(title or f"{dat_path.name} idx={index}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preview OHLC images")
    parser.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    parser.add_argument("--permno", type=int, default=None)
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOW_DAYS))
    parser.add_argument("--show", action="store_true", help="open matplotlib window")
    parser.add_argument(
        "--dat",
        type=Path,
        default=None,
        help="optional memmap .dat to preview instead of building from parquet",
    )
    parser.add_argument("--window-days", type=int, default=20)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    if args.dat is not None:
        fig = plot_memmap_index(args.dat, args.window_days, args.index)
    else:
        permno = args.permno if args.permno is not None else default_permno(args.parquet)
        samples = build_sample_windows(
            parquet_path=args.parquet,
            permno=permno,
            window_days_list=tuple(args.windows),
        )
        for window_days, _img, meta, as_of in samples:
            print(f"window={window_days} as_of={as_of} {meta}")
        fig = plot_sample_windows(samples, permno)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        print("built preview figure (pass --show to display)")


if __name__ == "__main__":
    main()
